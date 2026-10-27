import { createServer, type IncomingMessage, type Server } from "node:http";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createTokenSource,
  type FederationConfig,
  type FederationGateway,
  readFederationConfigs,
  startFederation,
  startFederationGateway,
} from "./federation.ts";

const ANTHROPIC_ENV = {
  ANTHROPIC_FEDERATION_RULE_ID: "fdrl_test",
  ANTHROPIC_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000000",
  ANTHROPIC_SERVICE_ACCOUNT_ID: "svac_test",
};

const OPENAI_ENV = {
  OPENAI_IDENTITY_PROVIDER_ID: "idp_test",
  OPENAI_SERVICE_ACCOUNT_ID: "sa_test",
  OPENAI_WIF_AUDIENCE: "https://api.openai.com/v1",
};

function anthropicConfig(overrides: Partial<FederationConfig> = {}): FederationConfig {
  const [config] = readFederationConfigs({ ...ANTHROPIC_ENV });
  if (!config) throw new Error("expected a config");
  return { ...config, ...overrides };
}

const servers: Server[] = [];
const gateways: FederationGateway[] = [];

afterEach(async () => {
  vi.unstubAllGlobals();
  await Promise.all(gateways.splice(0).map((gateway) => gateway.close()));
  await Promise.all(
    servers.splice(0).map(
      (server) =>
        new Promise<void>((resolve) => {
          server.closeAllConnections();
          server.close(() => resolve());
        })
    )
  );
});

async function listen(
  handler: (req: IncomingMessage, res: import("node:http").ServerResponse) => void
): Promise<string> {
  const server = createServer(handler);
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("no address");
  return `http://127.0.0.1:${address.port}`;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("readFederationConfigs", () => {
  it("returns nothing when no federation var is set", () => {
    expect(readFederationConfigs({})).toEqual([]);
  });

  it("refuses a partial configuration instead of silently skipping it", () => {
    expect(() =>
      readFederationConfigs({ ANTHROPIC_FEDERATION_RULE_ID: "fdrl_test" })
    ).toThrowError(/ANTHROPIC_ORGANIZATION_ID, ANTHROPIC_SERVICE_ACCOUNT_ID/);
    expect(() =>
      readFederationConfigs({ OPENAI_IDENTITY_PROVIDER_ID: "idp", OPENAI_SERVICE_ACCOUNT_ID: "sa" })
    ).toThrowError(/OPENAI_WIF_AUDIENCE/);
  });

  it("builds Anthropic's jwt-bearer exchange body", () => {
    const [config] = readFederationConfigs({ ...ANTHROPIC_ENV, ANTHROPIC_WORKSPACE_ID: "wrkspc_x" });
    expect(config?.tokenUrl).toBe("https://api.anthropic.com/v1/oauth/token");
    expect(config?.audience).toBe("https://api.anthropic.com");
    expect(config?.exchangeBody("jwt")).toEqual({
      grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
      assertion: "jwt",
      federation_rule_id: "fdrl_test",
      organization_id: "00000000-0000-0000-0000-000000000000",
      service_account_id: "svac_test",
      workspace_id: "wrkspc_x",
    });
  });

  it("omits workspace_id for single-workspace Anthropic rules", () => {
    const [config] = readFederationConfigs({ ...ANTHROPIC_ENV });
    expect(config?.exchangeBody("jwt")).not.toHaveProperty("workspace_id");
  });

  it("builds OpenAI's RFC 8693 token-exchange body", () => {
    const [config] = readFederationConfigs({ ...OPENAI_ENV });
    expect(config?.tokenUrl).toBe("https://auth.openai.com/oauth/token");
    expect(config?.audience).toBe("https://api.openai.com/v1");
    expect(config?.exchangeBody("jwt")).toEqual({
      grant_type: "urn:ietf:params:oauth:grant-type:token-exchange",
      subject_token_type: "urn:ietf:params:oauth:token-type:jwt",
      subject_token: "jwt",
      identity_provider_id: "idp_test",
      service_account_id: "sa_test",
    });
  });
});

describe("token source", () => {
  it("caches a token, then re-exchanges with a fresh JWT before it expires", async () => {
    let clock = 0;
    let minted = 0;
    const assertions: string[] = [];
    const fetchImpl = vi.fn(async (_url: unknown, init?: RequestInit) => {
      assertions.push(JSON.parse(String(init?.body)).assertion);
      return jsonResponse({ access_token: `tok_${assertions.length}`, expires_in: 600 });
    });
    const tokens = createTokenSource({
      config: anthropicConfig(),
      mintJwt: async () => `jwt_${++minted}`,
      fetchImpl: fetchImpl as unknown as typeof fetch,
      now: () => clock,
    });

    expect(await tokens.get()).toBe("tok_1");
    clock = 400_000;
    expect(await tokens.get()).toBe("tok_1");
    // inside the 120s refresh margin
    clock = 481_000;
    expect(await tokens.get()).toBe("tok_2");
    // jti is single-use: every exchange must present a different JWT
    expect(assertions).toEqual(["jwt_1", "jwt_2"]);
  });

  it("refreshes at half-life for short-lived tokens", async () => {
    let clock = 0;
    let exchanges = 0;
    const tokens = createTokenSource({
      config: anthropicConfig(),
      mintJwt: async () => "jwt",
      fetchImpl: (async () =>
        jsonResponse({ access_token: `tok_${++exchanges}`, expires_in: 120 })) as typeof fetch,
      now: () => clock,
    });
    await tokens.get();
    clock = 59_000;
    expect(await tokens.get()).toBe("tok_1");
    clock = 61_000;
    expect(await tokens.get()).toBe("tok_2");
  });

  it("shares one exchange between concurrent callers", async () => {
    let exchanges = 0;
    const tokens = createTokenSource({
      config: anthropicConfig(),
      mintJwt: async () => "jwt",
      fetchImpl: (async () => {
        exchanges++;
        await new Promise((resolve) => setTimeout(resolve, 20));
        return jsonResponse({ access_token: "tok", expires_in: 600 });
      }) as typeof fetch,
    });
    const results = await Promise.all([tokens.get(), tokens.get(), tokens.get()]);
    expect(results).toEqual(["tok", "tok", "tok"]);
    expect(exchanges).toBe(1);
  });

  it("names the provider and audience when the rule rejects the JWT", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({ error: { message: "Authentication failed" } }, 401)
    );
    const tokens = createTokenSource({
      config: anthropicConfig(),
      mintJwt: async () => "jwt",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await expect(tokens.get()).rejects.toThrowError(
      /Anthropic token exchange was rejected \(401\).*https:\/\/api\.anthropic\.com/
    );
    // a rejected rule is not transient, so it is not retried
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});

describe("federation gateway", () => {
  async function startGateway(upstream: string) {
    const gateway = await startFederationGateway({
      config: anthropicConfig({ upstream }),
      tokens: { get: async () => "sk-ant-oat01-minted" },
      clientKey: "placeholder",
    });
    gateways.push(gateway);
    return gateway;
  }

  it("refuses a request that doesn't present the run's placeholder", async () => {
    const upstream = await listen((_req, res) => res.end("should not be reached"));
    const gateway = await startGateway(upstream);
    const res = await fetch(`${gateway.url}/v1/messages`, {
      method: "POST",
      headers: { "x-api-key": "someone-elses-key" },
      body: "{}",
    });
    expect(res.status).toBe(401);
  });

  it("swaps the placeholder for the minted bearer token and forwards the request", async () => {
    let seen: {
      url?: string | undefined;
      headers?: IncomingMessage["headers"];
      body?: string;
    } = {};
    const upstream = await listen((req, res) => {
      let body = "";
      req.on("data", (chunk) => {
        body += chunk;
      });
      req.on("end", () => {
        seen = { url: req.url, headers: req.headers, body };
        res.writeHead(200, { "content-type": "application/json" });
        res.end('{"ok":true}');
      });
    });
    const gateway = await startGateway(upstream);

    const res = await fetch(`${gateway.url}/v1/messages?beta=true`, {
      method: "POST",
      headers: {
        "x-api-key": "placeholder",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: '{"model":"claude"}',
    });

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
    expect(seen.url).toBe("/v1/messages?beta=true");
    expect(seen.body).toBe('{"model":"claude"}');
    expect(seen.headers?.authorization).toBe("Bearer sk-ant-oat01-minted");
    expect(seen.headers?.["x-api-key"]).toBeUndefined();
    expect(seen.headers?.["anthropic-version"]).toBe("2023-06-01");
  });

  it("accepts the placeholder as a bearer token (OpenAI-style clients)", async () => {
    const upstream = await listen((req, res) => res.end(req.headers.authorization));
    const gateway = await startGateway(upstream);
    const res = await fetch(`${gateway.url}/v1/responses`, {
      method: "POST",
      headers: { authorization: "Bearer placeholder" },
      body: "{}",
    });
    expect(await res.text()).toBe("Bearer sk-ant-oat01-minted");
  });

  it("streams server-sent events as they arrive", async () => {
    let finish: (() => void) | undefined;
    const upstream = await listen((_req, res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write("data: first\n\n");
      finish = () => res.end("data: last\n\n");
    });
    const gateway = await startGateway(upstream);

    const res = await fetch(`${gateway.url}/v1/messages`, {
      method: "POST",
      headers: { "x-api-key": "placeholder" },
      body: "{}",
    });
    const reader = res.body?.getReader();
    if (!reader) throw new Error("no body");
    const first = await reader.read();
    // the upstream response is still open, so this chunk was not buffered
    expect(new TextDecoder().decode(first.value)).toBe("data: first\n\n");
    finish?.();
    let rest = "";
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      rest += new TextDecoder().decode(chunk.value);
    }
    expect(rest).toBe("data: last\n\n");
  });

  it("answers 502 with the exchange error when no token can be minted", async () => {
    const gateway = await startFederationGateway({
      config: anthropicConfig({ upstream: "http://127.0.0.1:1" }),
      tokens: {
        get: async () => {
          throw new Error("Anthropic token exchange was rejected (401)");
        },
      },
      clientKey: "placeholder",
    });
    gateways.push(gateway);
    const res = await fetch(`${gateway.url}/v1/messages`, {
      method: "POST",
      headers: { "x-api-key": "placeholder" },
      body: "{}",
    });
    expect(res.status).toBe(502);
    expect(await res.text()).toContain("rejected (401)");
  });
});

describe("startFederation", () => {
  const oidc = { requestUrl: "https://gha.example/token?api-version=2", requestToken: "req" };

  function stubProviders() {
    const audiences: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL) => {
        const url = new URL(String(input));
        if (url.hostname === "gha.example") {
          audiences.push(url.searchParams.get("audience") ?? "");
          return jsonResponse({ value: `jwt_${audiences.length}` });
        }
        return jsonResponse({ access_token: "minted", expires_in: 600 });
      })
    );
    return audiences;
  }

  it("does nothing when federation isn't configured", async () => {
    expect(await startFederation({ oidc, env: {} })).toBeNull();
  });

  it("requires the run's OIDC credentials", async () => {
    await expect(startFederation({ oidc: null, env: { ...OPENAI_ENV } })).rejects.toThrowError(
      /id-token: write/
    );
  });

  it("refuses to stack on a customer gateway", async () => {
    stubProviders();
    await expect(
      startFederation({
        oidc,
        env: { ...ANTHROPIC_ENV, ANTHROPIC_BASE_URL: "https://gateway.example" },
      })
    ).rejects.toThrowError(/ANTHROPIC_BASE_URL/);
  });

  it("points each harness at its gateway and displaces competing credentials", async () => {
    const audiences = stubProviders();
    const env: NodeJS.ProcessEnv = {
      ...ANTHROPIC_ENV,
      ...OPENAI_ENV,
      ANTHROPIC_API_KEY: "sk-ant-static",
      CLAUDE_CODE_OAUTH_TOKEN: "subscription",
      CODEX_AUTH_JSON: "{}",
    };
    const federation = await startFederation({ oidc, env });
    if (!federation) throw new Error("expected federation");
    try {
      expect(federation.providers).toEqual(["anthropic", "openai"]);
      expect(audiences).toEqual(["https://api.anthropic.com", "https://api.openai.com/v1"]);
      expect(env.ANTHROPIC_API_KEY).toMatch(/^pullfrog-wif-[0-9a-f]{48}$/);
      expect(env.OPENAI_API_KEY).toMatch(/^pullfrog-wif-[0-9a-f]{48}$/);
      expect(env.ANTHROPIC_BASE_URL).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
      expect(env.OPENAI_BASE_URL).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/v1$/);
      expect(env.CLAUDE_CODE_OAUTH_TOKEN).toBeUndefined();
      expect(env.CODEX_AUTH_JSON).toBeUndefined();
      expect(federation.suppressedCredentials).toEqual([
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CODEX_AUTH_JSON",
      ]);
    } finally {
      await federation[Symbol.asyncDispose]();
    }
  });
});

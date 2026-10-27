/**
 * Workload identity federation (keyless OIDC auth) for Anthropic and OpenAI.
 *
 * Both providers can exchange a GitHub Actions OIDC JWT for a short-lived API
 * token, but neither token survives a run: Anthropic caps it at twice the JWT's
 * remaining life (~10 min for a fresh GitHub token) and OpenAI never lets it
 * outlive the JWT at all (~5 min). A one-shot exchange into `ANTHROPIC_API_KEY`
 * would 401 a few minutes into any real run, and only claude-code has a refresh
 * hook (`apiKeyHelper`) — codex and opencode read the key once at spawn.
 *
 * So each federated provider gets a localhost gateway instead, plugged into the
 * existing `PROVIDER_GATEWAY_URL_ENV` support that all three harnesses already
 * honour. The agent is handed `<PROVIDER>_BASE_URL` pointing here plus a random
 * per-run placeholder key; the gateway checks that key, swaps it for a token it
 * re-exchanges before expiry, and streams the request to the real API. The
 * minted token never enters any agent's env, and the placeholder is worthless
 * once this process exits.
 *
 * Every exchange mints a FRESH GitHub JWT from the stashed OIDC credentials:
 * Anthropic treats a JWT's `jti` as single-use by default (`jti_reused`), so
 * the 60s-cached `mintIdToken` must never feed this.
 *
 * Configured from the workflow `env:` block with the providers' own variable
 * names, so the console wizard output pastes straight in:
 *   - Anthropic: ANTHROPIC_FEDERATION_RULE_ID, ANTHROPIC_ORGANIZATION_ID,
 *     ANTHROPIC_SERVICE_ACCOUNT_ID, ANTHROPIC_WORKSPACE_ID (multi-workspace
 *     rules only), ANTHROPIC_WIF_AUDIENCE (default https://api.anthropic.com)
 *   - OpenAI: OPENAI_IDENTITY_PROVIDER_ID, OPENAI_SERVICE_ACCOUNT_ID,
 *     OPENAI_WIF_AUDIENCE (required — OpenAI publishes no default)
 */

import { randomBytes } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { Readable } from "node:stream";
import * as yes from "yes";
import { log } from "./cli.ts";
import { fetchIdTokenFromStash, isTransientTokenError, type OidcCredentials } from "./github.ts";
import { maskSecret } from "./secretCommands.ts";

export type FederationProvider = "anthropic" | "openai";

export interface FederationConfig {
  provider: FederationProvider;
  displayName: string;
  /** the `aud` the GitHub JWT is minted for; must match the provider-side rule. */
  audience: string;
  /** real API origin the gateway forwards to. */
  upstream: string;
  tokenUrl: string;
  /** exchange request body for a given GitHub JWT. */
  exchangeBody: (jwt: string) => Record<string, string>;
  /** env var the agent reads its credential from; receives the placeholder. */
  keyEnv: string;
  /** env var that re-points every harness at the gateway (PROVIDER_GATEWAY_URL_ENV). */
  baseUrlEnv: string;
  /** appended to the gateway origin to match the base-URL env var's convention. */
  baseUrlPath: string;
  /** credentials that would outrank or bypass the placeholder, cleared when federating. */
  displacedEnv: string[];
}

const ANTHROPIC_REQUIRED = [
  "ANTHROPIC_FEDERATION_RULE_ID",
  "ANTHROPIC_ORGANIZATION_ID",
  "ANTHROPIC_SERVICE_ACCOUNT_ID",
] as const;

const OPENAI_REQUIRED = [
  "OPENAI_IDENTITY_PROVIDER_ID",
  "OPENAI_SERVICE_ACCOUNT_ID",
  "OPENAI_WIF_AUDIENCE",
] as const;

function read(env: NodeJS.ProcessEnv, name: string): string | undefined {
  return env[name]?.trim() || undefined;
}

/** a provider is configured as soon as any of its required vars is set — a partial set is an error, never a silent skip. */
function requireAll(
  env: NodeJS.ProcessEnv,
  displayName: string,
  names: readonly string[]
): Record<string, string> | null {
  const present = names.filter((name) => read(env, name));
  if (present.length === 0) return null;
  const missing = names.filter((name) => !read(env, name));
  if (missing.length > 0) {
    throw new Error(
      `${displayName} workload identity federation is partly configured: set ${missing.join(", ")} ` +
        `alongside ${present.join(", ")} in the workflow's env: block`
    );
  }
  return Object.fromEntries(names.map((name) => [name, read(env, name) as string]));
}

export function readFederationConfigs(env: NodeJS.ProcessEnv = process.env): FederationConfig[] {
  const configs: FederationConfig[] = [];

  const anthropic = requireAll(env, "Anthropic", ANTHROPIC_REQUIRED);
  if (anthropic) {
    const workspaceId = read(env, "ANTHROPIC_WORKSPACE_ID");
    configs.push({
      provider: "anthropic",
      displayName: "Anthropic",
      audience: read(env, "ANTHROPIC_WIF_AUDIENCE") ?? "https://api.anthropic.com",
      upstream: "https://api.anthropic.com",
      tokenUrl: "https://api.anthropic.com/v1/oauth/token",
      exchangeBody: (jwt) => ({
        grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
        assertion: jwt,
        federation_rule_id: anthropic.ANTHROPIC_FEDERATION_RULE_ID as string,
        organization_id: anthropic.ANTHROPIC_ORGANIZATION_ID as string,
        service_account_id: anthropic.ANTHROPIC_SERVICE_ACCOUNT_ID as string,
        ...(workspaceId ? { workspace_id: workspaceId } : {}),
      }),
      keyEnv: "ANTHROPIC_API_KEY",
      // ANTHROPIC_BASE_URL is a bare host: claude-code and the SDKs append /v1 themselves
      baseUrlEnv: "ANTHROPIC_BASE_URL",
      baseUrlPath: "",
      // claude-code ranks ANTHROPIC_AUTH_TOKEN above ANTHROPIC_API_KEY
      displacedEnv: ["ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"],
    });
  }

  const openai = requireAll(env, "OpenAI", OPENAI_REQUIRED);
  if (openai) {
    configs.push({
      provider: "openai",
      displayName: "OpenAI",
      audience: openai.OPENAI_WIF_AUDIENCE as string,
      upstream: "https://api.openai.com",
      tokenUrl: "https://auth.openai.com/oauth/token",
      exchangeBody: (jwt) => ({
        grant_type: "urn:ietf:params:oauth:grant-type:token-exchange",
        subject_token_type: "urn:ietf:params:oauth:token-type:jwt",
        subject_token: jwt,
        identity_provider_id: openai.OPENAI_IDENTITY_PROVIDER_ID as string,
        service_account_id: openai.OPENAI_SERVICE_ACCOUNT_ID as string,
      }),
      keyEnv: "OPENAI_API_KEY",
      // the OPENAI_BASE_URL convention includes /v1
      baseUrlEnv: "OPENAI_BASE_URL",
      baseUrlPath: "/v1",
      // a ChatGPT subscription outranks OPENAI_API_KEY in both codex and opencode
      displacedEnv: ["CODEX_AUTH_JSON"],
    });
  }

  return configs;
}

/** status-bearing so the retry predicate can tell an outage from a rejected rule. */
export class FederationExchangeError extends Error {
  readonly status: number | undefined;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "FederationExchangeError";
    this.status = status;
  }
}

function isTransientExchangeError(error: unknown): boolean {
  if (error instanceof FederationExchangeError) {
    return error.status === undefined || error.status >= 500 || error.status === 429;
  }
  return isTransientTokenError(error);
}

/** refresh this long before expiry — capped at half the lifetime, since an OpenAI token may only live ~5 min. */
const REFRESH_MARGIN_MS = 120_000;

export interface TokenSource {
  get: () => Promise<string>;
}

export function createTokenSource(params: {
  config: FederationConfig;
  /** returns a freshly minted GitHub JWT for the given audience — never a cached one. */
  mintJwt: (audience: string) => Promise<string>;
  fetchImpl?: typeof fetch;
  now?: () => number;
}): TokenSource {
  const { config } = params;
  const fetchImpl = params.fetchImpl ?? fetch;
  const now = params.now ?? Date.now;
  let cached: { token: string; refreshAt: number } | undefined;
  let inflight: Promise<string> | undefined;

  const exchangeOnce = async (): Promise<string> => {
    const jwt = await params.mintJwt(config.audience);
    let response: Response;
    try {
      response = await fetchImpl(config.tokenUrl, {
        method: "POST",
        headers: { "content-type": "application/json", accept: "application/json" },
        body: JSON.stringify(config.exchangeBody(jwt)),
        signal: AbortSignal.timeout(30_000),
      });
    } catch (error) {
      throw new FederationExchangeError(
        `${config.displayName} token exchange request failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
    if (!response.ok) {
      // both providers answer a rejected rule with an opaque 401; the real reason
      // lives in their console's authentication history, so say where to look.
      const detail = (await response.text().catch(() => "")).slice(0, 300);
      throw new FederationExchangeError(
        `${config.displayName} token exchange was rejected (${response.status})${detail ? `: ${detail}` : ""}. ` +
          `check the federation rule's audience (${config.audience}) and subject match against this workflow run`,
        response.status
      );
    }
    const body = (await response.json()) as { access_token?: string; expires_in?: number };
    if (!body.access_token) {
      throw new FederationExchangeError(`${config.displayName} token exchange returned no access_token`);
    }
    maskSecret(body.access_token);
    const lifetimeMs = Math.max(0, (body.expires_in ?? 300) * 1000);
    cached = {
      token: body.access_token,
      refreshAt: now() + lifetimeMs - Math.min(REFRESH_MARGIN_MS, lifetimeMs / 2),
    };
    return body.access_token;
  };

  const exchange = yes.mutation({
    run: exchangeOnce,
    name: `${config.provider} federation exchange`,
    retry: (error, attempt) =>
      isTransientExchangeError(error) ? yes.backoff([1000, 2000], attempt) : -1,
  });

  return {
    get: () => {
      if (cached && now() < cached.refreshAt) return Promise.resolve(cached.token);
      // one exchange at a time: concurrent requests during a refresh share it
      inflight ??= exchange().finally(() => {
        inflight = undefined;
      });
      return inflight;
    },
  };
}

// hop-by-hop or recomputed by fetch; never forwarded in either direction
const DROPPED_REQUEST_HEADERS = new Set([
  "host",
  "connection",
  "keep-alive",
  "transfer-encoding",
  "content-length",
  "upgrade",
  "proxy-authorization",
  // fetch would transparently decompress, so ask upstream for identity bytes
  "accept-encoding",
  "x-api-key",
  "authorization",
]);

const DROPPED_RESPONSE_HEADERS = new Set([
  "connection",
  "keep-alive",
  "transfer-encoding",
  "content-length",
  "content-encoding",
]);

function presentedKey(req: IncomingMessage): string | undefined {
  const apiKey = req.headers["x-api-key"];
  if (typeof apiKey === "string") return apiKey;
  const auth = req.headers.authorization;
  if (typeof auth === "string" && auth.toLowerCase().startsWith("bearer ")) return auth.slice(7);
  return undefined;
}

function sendError(res: ServerResponse, status: number, message: string): void {
  if (res.headersSent) {
    res.destroy();
    return;
  }
  res.writeHead(status, { "content-type": "application/json" });
  // the Anthropic error envelope; OpenAI clients read `error.message` from it too
  res.end(JSON.stringify({ type: "error", error: { type: "api_error", message } }));
}

async function readBody(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks);
}

export interface FederationGateway {
  url: string;
  close: () => Promise<void>;
}

export async function startFederationGateway(params: {
  config: FederationConfig;
  tokens: TokenSource;
  /** the placeholder key the agent presents; anything else is refused. */
  clientKey: string;
  fetchImpl?: typeof fetch;
}): Promise<FederationGateway> {
  const { config, tokens, clientKey } = params;
  const fetchImpl = params.fetchImpl ?? fetch;

  const server = createServer(async (req, res) => {
    // localhost is reachable from the agent's own shell too, so the gateway is
    // no more open than a key in the agent's env would be — but no less closed.
    if (presentedKey(req) !== clientKey) {
      sendError(res, 401, "pullfrog federation gateway: unrecognized credential");
      return;
    }
    if (!req.url?.startsWith("/")) {
      sendError(res, 400, "pullfrog federation gateway: bad request path");
      return;
    }

    let token: string;
    try {
      token = await tokens.get();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      log.warning(`» ${config.displayName} federation: ${message}`);
      sendError(res, 502, `pullfrog federation gateway: ${message}`);
      return;
    }

    const headers = new Headers();
    for (const [name, value] of Object.entries(req.headers)) {
      if (value === undefined || DROPPED_REQUEST_HEADERS.has(name)) continue;
      headers.set(name, Array.isArray(value) ? value.join(", ") : value);
    }
    headers.set("authorization", `Bearer ${token}`);

    const abort = new AbortController();
    res.on("close", () => {
      if (!res.writableFinished) abort.abort();
    });

    try {
      const method = req.method ?? "GET";
      const body = method === "GET" || method === "HEAD" ? undefined : await readBody(req);
      const upstream = await fetchImpl(new URL(req.url, config.upstream), {
        method,
        headers,
        ...(body ? { body: new Uint8Array(body) } : {}),
        redirect: "manual",
        signal: abort.signal,
      });
      const responseHeaders: Record<string, string> = {};
      upstream.headers.forEach((value, name) => {
        if (!DROPPED_RESPONSE_HEADERS.has(name)) responseHeaders[name] = value;
      });
      res.writeHead(upstream.status, responseHeaders);
      if (!upstream.body) {
        res.end();
        return;
      }
      // streamed, not buffered: SSE responses must reach the agent as they arrive
      // pipe() doesn't forward source errors, and an upstream that dies mid-stream
      // would otherwise leave the agent waiting out its own timeout
      Readable.fromWeb(upstream.body as import("node:stream/web").ReadableStream)
        .on("error", () => res.destroy())
        .pipe(res);
    } catch (error) {
      if (abort.signal.aborted) return;
      sendError(
        res,
        502,
        `pullfrog federation gateway: upstream request failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("federation gateway failed to bind");
  }

  return {
    url: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

export interface Federation {
  providers: FederationProvider[];
  /** env vars this run now owns; main drops them from the db-injected list. */
  managedEnv: string[];
  /** subscription credentials the pool must not load back in. */
  suppressedCredentials: string[];
  [Symbol.asyncDispose]: () => Promise<void>;
}

/**
 * Exchange once up front, so a misconfigured rule fails the run before any
 * agent spawns with a message naming the provider, then hand each harness its
 * gateway. Returns null when no provider is configured.
 */
export async function startFederation(params: {
  oidc: OidcCredentials | null;
  env?: NodeJS.ProcessEnv;
}): Promise<Federation | null> {
  const env = params.env ?? process.env;
  const configs = readFederationConfigs(env);
  if (configs.length === 0) return null;

  const oidc = params.oidc;
  if (!oidc) {
    throw new Error(
      "workload identity federation needs a GitHub Actions OIDC token: run Pullfrog in GitHub Actions " +
        "with `permissions: id-token: write`"
    );
  }

  const gateways: FederationGateway[] = [];
  const federation: Federation = {
    providers: [],
    managedEnv: [],
    suppressedCredentials: [],
    [Symbol.asyncDispose]: async () => {
      await Promise.all(gateways.map((gateway) => gateway.close()));
    },
  };

  try {
    for (const config of configs) {
      if (read(env, config.baseUrlEnv)) {
        throw new Error(
          `${config.baseUrlEnv} and ${config.displayName} workload identity federation can't be combined: ` +
            `federation talks to ${config.upstream} directly. remove one of them from the workflow`
        );
      }

      const tokens = createTokenSource({
        config,
        mintJwt: (audience) => fetchIdTokenFromStash(oidc, audience),
      });
      await tokens.get();

      const clientKey = `pullfrog-wif-${randomBytes(24).toString("hex")}`;
      maskSecret(clientKey);
      const gateway = await startFederationGateway({ config, tokens, clientKey });
      gateways.push(gateway);

      const ignored = [config.keyEnv, ...config.displacedEnv].filter((name) => read(env, name));
      if (ignored.length > 0) {
        log.warning(
          `» ${config.displayName}: ignoring ${ignored.join(", ")} — workload identity federation is configured and takes precedence`
        );
      }
      for (const name of config.displacedEnv) delete env[name];
      env[config.keyEnv] = clientKey;
      env[config.baseUrlEnv] = `${gateway.url}${config.baseUrlPath}`;

      federation.providers.push(config.provider);
      federation.managedEnv.push(config.keyEnv, config.baseUrlEnv);
      federation.suppressedCredentials.push(...config.displacedEnv);
      log.info(`» ${config.displayName}: authenticating with workload identity federation (OIDC)`);
    }
  } catch (error) {
    await federation[Symbol.asyncDispose]();
    throw error;
  }

  return federation;
}

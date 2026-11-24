/** Small, dependency-free client for the native Loci HTTP API. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export interface LociDocument {
  id: string;
  revision: string;
  text: string;
  fields?: Record<string, Json>;
  readers?: string[];
  /** False keeps the document lexical on an index that otherwise embeds writes. */
  embed?: boolean;
}

export type Mutation =
  | { op: "upsert"; document: LociDocument }
  | { op: "delete"; id: string };

export interface IndexSettings {
  model?: "lexical" | "bge-small-en-v1.5" | "multilingual-e5-small";
  mappings?: Record<string, string>;
  default_visibility?: "public" | "private";
  stream?: {
    timestamp: string;
    rollover_seconds: number;
    retention_seconds: number;
    automatic?: boolean;
  };
}

export interface ExactPage {
  generation: number;
  total: number;
  hits: LociDocument[];
  facets: Record<string, { value: Json; count: number }[]>;
  aggregations: Record<string, Json>;
  next_cursor?: string | null;
}

export interface RegionalHit {
  id: string;
  document: string;
  revision: string;
  start: number;
  end: number;
  text: string;
  score: number;
}

export interface RegionalPage {
  generation: number;
  hits: RegionalHit[];
}

export class LociError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "LociError";
    this.status = status;
  }
}

export class Loci {
  private readonly baseUrl: string;
  private readonly token?: string;

  constructor(baseUrl = "http://127.0.0.1:7707", token?: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.token = token;
  }

  private async request<T>(method: string, path: string, body?: unknown, key?: string): Promise<T> {
    const headers = new Headers({ Accept: "application/json" });
    if (this.token) headers.set("Authorization", "Bearer " + this.token);
    if (key) headers.set("Idempotency-Key", key);
    if (body !== undefined) headers.set("Content-Type", "application/json");
    const response = await fetch(this.baseUrl + path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      redirect: "error",
    });
    if (!response.ok) {
      throw new LociError(response.status, await response.text());
    }
    return (await response.json()) as T;
  }

  createIndex(name: string, settings: IndexSettings = {}) {
    return this.request<{ name: string; created: boolean }>("PUT", this.index(name), settings);
  }

  bulk(name: string, operations: Mutation[], idempotencyKey?: string) {
    return this.request<{ generation: number; changed: number }>(
      "POST", this.index(name) + "/bulk", { operations }, idempotencyKey,
    );
  }

  search(name: string, query: Record<string, Json> & { mode: "exact" }): Promise<ExactPage>;
  search(name: string, query: Record<string, Json> & { mode: "regional" }): Promise<RegionalPage>;
  search(name: string, query: Record<string, Json>): Promise<ExactPage | RegionalPage>;
  search(name: string, query: Record<string, Json>): Promise<ExactPage | RegionalPage> {
    return this.request("POST", this.index(name) + "/search", query);
  }

  upgradeModel(name: string, model: "bge-small-en-v1.5" | "multilingual-e5-small") {
    return this.request<{ generation: number; model: string; documents: number }>(
      "PUT", this.index(name) + "/model", { model },
    );
  }

  private index(name: string) {
    return "/v1/indexes/" + encodeURIComponent(name);
  }
}

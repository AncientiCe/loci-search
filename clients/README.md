# Using the Loci clients

Start `loci serve` before running the HTTP examples. Both clients use the native versioned API and can supply a scoped bearer token. Pass the token as the second constructor argument when the service requires credentials. Install the TypeScript source alongside your application or import the Python module directly.

```typescript
import { Loci } from "./typescript/loci.ts";

const client = new Loci("http://127.0.0.1:7707");
await client.createIndex("manuals", { model: "lexical" });
await client.bulk("manuals", [{
  op: "upsert",
  document: { id: "recovery", revision: "1", text: "Restore the vault from its backup." },
}]);
const page = await client.search("manuals", { mode: "exact", text: "vault" });
console.log(page.hits);
```

Run the Python example with `clients/python` on the module search path.

```python
from loci import Loci

client = Loci()
client.create_index("manuals", {"model": "lexical"})
client.bulk("manuals", [{"op": "upsert", "document": {
    "id": "recovery", "revision": "1", "text": "Restore the vault from its backup.",
}}])
print(client.search("manuals", mode="exact", text="vault")["hits"])
```

To use the Rust library directly, run `cargo run --locked --example embedded`. The example creates a temporary local catalog, commits a document and reads its matching revision through a snapshot. The [HTTP reference](../docs/openapi.json) describes additional endpoints.

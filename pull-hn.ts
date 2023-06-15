import fs from "fs";
import readline from "readline";

const BASE_URL = "https://hacker-news.firebaseio.com/v0";
const BATCH_SIZE = 1000;

const outFile = "/workspace/data/hn/full_dump.jsonl";

const existingIds = new Set<number>();

// check if outfile exists
if (fs.existsSync(outFile)) {
  // Iterate over line by line and parse json
  for await (const line of readline.createInterface({
    input: fs.createReadStream(outFile),
    crlfDelay: Infinity,
  })) {
    try {
      const { id } = JSON.parse(line);
      existingIds.add(id);
    } catch (e) {
      console.error("Error parsing line", line, e);
    }
  }
}

const maxId: number = await fetch(`${BASE_URL}/maxitem.json`).then((res) => res.json());

console.log(
  `Already pulled ${existingIds.size} / ${maxId} items (${Math.round(
    (existingIds.size / maxId) * 100
  )}%)`
);

const remainingToPull = maxId - existingIds.size;

let totalProcessed = 0;
let nextId = maxId;
while (nextId > 0) {
  const batch: number[] = [];
  while (batch.length < BATCH_SIZE && nextId > 0) {
    if (!existingIds.has(nextId)) {
      batch.push(nextId);
    }
    nextId--;
  }
  const batchJson = (
    await Promise.all(
      batch.map(async (id) => {
        try {
          let json = await fetch(`${BASE_URL}/item/${id}.json`).then((res) => res.json());
          if (json === null) {
            json = { id, deleted: true };
          }

          return json;
        } catch (e) {
          console.error("Error fetching", id, e);
          return null;
          // return { id, error: true };
        }
      })
    )
  ).filter(Boolean);
  // Write the batch to the jsonl file
  fs.appendFileSync(outFile, batchJson.map((json) => JSON.stringify(json)).join("\n") + "\n");
  totalProcessed += batchJson.length;

  console.log(
    `Processed ${totalProcessed} of ${remainingToPull} (${
      Math.round((totalProcessed / remainingToPull) * 100 * 100) / 100
    }%)`
  );
}
// if (await fs.access(outFile)) {
//   const lines = await fs.readFile(outFile, "utf-8");
//   for (const line of lines.split("\n")) {
//     const { id } = JSON.parse(line);
//     existingIds.add(id);
//   }
// }

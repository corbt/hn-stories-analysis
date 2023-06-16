import fs from "fs";
import readline from "readline";

const inFile = "/workspace/data/hn/full_dump.jsonl";
const outFile = "/workspace/data/hn/stories_dump.jsonl";

// Clear the outFile
fs.writeFileSync(outFile, "");

let lineCount = 0;
let storiesCount = 0;

let storiesBuffer: string[] = [];

for await (const line of readline.createInterface({
  input: fs.createReadStream(inFile),
  crlfDelay: Infinity,
})) {
  lineCount++;

  if (lineCount % 100000 === 0) {
    console.log(`Processed ${lineCount} lines, ${storiesCount} stories`);
  }
  const { type } = JSON.parse(line);
  if (type === "story") {
    storiesBuffer.push(line);

    if (storiesBuffer.length >= 10000) {
      fs.appendFileSync(outFile, storiesBuffer.join("\n") + "\n");
      storiesBuffer = [];
    }
    storiesCount++;
  }
}

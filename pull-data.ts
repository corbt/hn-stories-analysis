import WebTorrent from "webtorrent";
import fetch from "fetch";
import fs from "fs/promises";

console.log("Starting client");

const client = new WebTorrent();

// Note that the torrent will never "finish" because we are not downloading the entire file. So you just have to
// kill the process when you are done.

// From https://academictorrents.com/details/c861d265525c488a9439fb874bd9c3fc38dcdfa5

const magnetURI =
  "magnet:?xt=urn:btih:c861d265525c488a9439fb874bd9c3fc38dcdfa5&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce";

client.on("error", (err) => {
  console.error("Client error:", err);
});

client.add(magnetURI, { path: "/workspace/data" }, (torrent) => {
  console.log("Client is downloading:", torrent.infoHash);

  // Super hacky workaround since the deselect method is actually broken upstream
  torrent._selections = [];
  torrent._updateSelections();
  console.log("Remaining selections:", torrent._selections);

  for (const file of torrent.files) {
    if (file.name.startsWith("RS")) {
      console.log("Selecting file:", file.name);
      file.select();
    } else {
      console.log("Deselecting file:", file.name);
      file.deselect();
    }
  }

  // Track progress. Overwrite each value with the updated, rounded, percentage.
  let last = 0;
  const interval = setInterval(() => {
    // Get the percentage for the file we are downloading

    for (const file of torrent.files) {
      console.log(`  ${file.name}: ${file.progress * 100}%`);
    }

    const percent = Math.round(torrent.progress * 100 * 100) / 100;
    const dlSpeed = Math.round((torrent.downloadSpeed / 1024 / 1024) * 100) / 100;
    const ulSpeed = Math.round((torrent.uploadSpeed / 1024 / 1024) * 100) / 100;
    const totalDownloaded = Math.round((torrent.downloaded / 1024 / 1024) * 100) / 100;
    if (percent !== last) {
      last = percent;
      console.log(
        `Downloaded ${totalDownloaded} MB (${percent}%) DL: ${dlSpeed} MB/s UL: ${ulSpeed} MB/s`
      );
    }
  }, 5000);

  torrent.on("done", () => {
    console.log("Torrent download finished");
    clearInterval(interval);
  });
});

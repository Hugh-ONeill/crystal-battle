#!/usr/bin/env node
// Batch team validator: reads newline-delimited team-file PATHS on stdin,
// validates each against a format in ONE node process (avoids ~300ms/team
// CLI startup), and emits one JSON result per line on stdout:
//   {"path": "...", "ok": true,  "species": ["...", x6]}
//   {"path": "...", "ok": false, "errors": ["...", ...]}
//
// Usage (from the pokemon-showdown checkout):
//   ls teamdir/*.team | node validate_teams.js gen9ou
//
// PS_DIST env overrides the checkout location (defaults to cwd).
"use strict";
const fs = require("fs");
const readline = require("readline");

const format = process.argv[2] || "gen9ou";
const dist = process.env.PS_DIST || process.cwd();
const { Teams } = require(dist + "/dist/sim/teams.js");
const { TeamValidator } = require(dist + "/dist/sim/team-validator.js");
const validator = new TeamValidator(format);

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (path) => {
  path = path.trim();
  if (!path) return;
  let out;
  try {
    const text = fs.readFileSync(path, "utf8");
    const team = Teams.import(text);
    if (!team || !team.length) {
      out = { path, ok: false, errors: ["unparseable / empty team"] };
    } else {
      const errs = validator.validateTeam(team);
      out = errs && errs.length
        ? { path, ok: false, errors: errs }
        : { path, ok: true, species: team.map((m) => m.species) };
    }
  } catch (e) {
    out = { path, ok: false, errors: ["exception: " + (e && e.message)] };
  }
  process.stdout.write(JSON.stringify(out) + "\n");
});

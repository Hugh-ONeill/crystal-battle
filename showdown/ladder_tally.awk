# ladder_tally.awk -- fast breakdown of ladder session logs (the `ladder tally`
# back end). Cheap counting pass only; showdown/loss_trace.py is the deep read.
#
# Games are keyed by BATTLE ROOM, never by the "=== game N ===" slot marker.
# A slot killed by PER_GAME_TIMEOUT leaves its battle live on the server and
# the next process rejoins it, so that game's result lands inside the FOLLOWING
# slot's block. Slot-keyed counting double-books one game and loses the other.
#
# Also splits the record by HOW a game ended. On a thin baseline queue a large
# share of "wins" are opponents timing out at team preview, which flatters the
# headline number badly -- those are reported separately, never silently.
#
# Usage:
#   awk -v me=PAC-Crystal [-v show_teams=1] -f ladder_tally.awk <logs...>

BEGIN {
    if (me == "") me = "PAC-Crystal"
    Z = 1.96
    # optional team->archetype map (showdown/classify_pool.py) for `arch` view
    if (archmap != "") {
        while ((getline aline < archmap) > 0)
            if (split(aline, aa, "\t") >= 2) { arch_of[aa[1]] = aa[2]; have_arch = 1 }
        close(archmap)
    }
}

# ---- per-file bookkeeping -------------------------------------------------
FNR == 1 {
    cur_file = FILENAME
    sub(/.*\//, "", cur_file)
    files[++nfiles] = cur_file
    cur_team = "-"
}

# ---- slot markers (written by ladder_session.sh) --------------------------
/^=== game [0-9]+\/[0-9]+ team: / {
    slots[cur_file]++; cur_team = $5
    # Fold weighting duplicates into the team they copy. A subpool was weighted
    # by making N byte-identical copies (`ah1_x`, `ah1_x_v2`, `ah1_x_v3`) since
    # the rotation draws with `ls | shuf`; the copies were deleted when that
    # weighting came out (2026-07-31), so their historical games had no file
    # left to classify and fell into the "?" archetype. They are the SAME team,
    # so their record belongs on the stem — both here and in the per-team rows.
    # No real pool name ends in _v<digits> (they end in a core hash or a word).
    sub(/_v[0-9]+$/, "", cur_team)
    # same idea for teams RENAMED in place: `40_greattusk_bootsbal` became
    # `ah1_greattusk_bootsbal` when the subpool got its prefix, and its earlier
    # games would otherwise stand as a separate team that no longer exists
    if (cur_team == "40_greattusk_bootsbal") cur_team = "ah1_greattusk_bootsbal"
    next
}
/^=== game [0-9]+ TIMED OUT/      { killed[cur_file]++; next }

# ---- config banner --------------------------------------------------------
# "=== session config | commit=X | format=Y | ... | argv= <rest>". Sessions
# weeks apart are only comparable if you can see what each one ran with;
# logs written before 2026-07-28 have no banner and read as "-".
/^=== session config \|/ {
    if (match($0, /commit=[^ |]+/)) commit[cur_file] = substr($0, RSTART + 7, RLENGTH - 7)
    if (match($0, /\| argv= /)) {
        a = substr($0, RSTART + RLENGTH)
        gsub(/--desk-log +[^ ]+ */, "", a)     # per-session by construction, not config
        sub(/[ \t]+$/, "", a)
        argv[cur_file] = a
    }
    next
}

# ---- room tracking: last battle id on the line wins ----------------------
# covers "<<< >battle-x", ">>> battle-x|/choose", "|/leave battle-x" and the
# |updatesearch| JSON. Protocol continuation lines inherit the last room.
{
    line = $0; rr = ""
    while (match(line, /battle-[a-z0-9]+-[0-9]+/)) {
        rr = substr(line, RSTART, RLENGTH)
        line = substr(line, RSTART + RLENGTH)
    }
    if (rr != "") {
        if (!(rr in seen)) {
            seen[rr] = 1
            order[++nrooms] = rr
            team[rr] = cur_team
            src[rr] = cur_file
            how[rr] = "board"
        }
        room = rr
    }
}

/^\|player\|p[12]\|/ {
    if (room == "") next
    split($0, f, "|")
    if (f[4] != "" && f[4] != me) opp[room] = f[4]
    next
}

/^\|turn\|[0-9]+/ {
    if (room == "") next
    split($0, f, "|")
    if (f[3] + 0 > turns[room]) turns[room] = f[3] + 0
    next
}

# clock forfeits: "|-message|NAME lost due to inactivity."
/^\|-message\|.* lost due to inactivity\./ {
    if (room == "") next
    who = $0
    sub(/^\|-message\|/, "", who)
    sub(/ lost due to inactivity\..*$/, "", who)
    how[room] = (who == me ? "our_clock" : "opp_clock")
    next
}
/^\|-message\|All players are inactive\./ { if (room != "") how[room] = "both_clock"; next }

/^\|win\|/ {
    if (room == "") next
    w = substr($0, 6); sub(/[ \t\r]+$/, "", w)
    res[room] = (w == me ? "W" : "L")
    next
}
/^\|tie[ \t\r]*$/ { if (room != "") res[room] = "T"; next }

# ---- session wall clock ---------------------------------------------------
/^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]/ {
    ts = $1 " " $2
    sub(/,[0-9]+$/, "", ts)
    if (!(cur_file in t0)) t0[cur_file] = ts
    t1[cur_file] = ts
}

# ---------------------------------------------------------------------------
function epoch(ts,   d) {          # "YYYY-MM-DD HH:MM:SS" -> seconds
    d = ts
    gsub(/[-:]/, " ", d)
    return mktime(d)
}
function dur(a, b,   s) {
    if (a == "" || b == "") return "?"
    s = epoch(b) - epoch(a)
    if (s < 0) return "?"
    return sprintf("%dh%02dm", int(s / 3600), int(s % 3600 / 60))
}
function rate(w, l) { return (w + l) ? sprintf("%5.1f%%", 100 * w / (w + l)) : "    -" }
function rec(w, l, t) { return sprintf("%dW-%dL%s", w + 0, l + 0, (t ? "-" t "T" : "")) }
function cfg(f_,   c) {             # "commit  argv" as recorded by the banner
    c = (commit[f_] == "" ? "-" : commit[f_])
    return c (argv[f_] == "" ? "" : "  " argv[f_])
}
function wilson(w, l,   n, p, c, h) {   # 95% Wilson interval, honest at small n
    n = w + l
    if (n == 0) return ""
    p = w / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return sprintf("[%.0f%%, %.0f%%]", 100 * (c - h), 100 * (c + h))
}
function leadname(t,   a) {         # "02_pelipper_debe17fc" -> "pelipper"
    a = t; sub(/^[0-9]+_/, "", a); sub(/_[0-9a-f]+$/, "", a); return a
}
function bar(w, l,   n, k, s) {     # 12-cell winrate bar
    n = w + l
    if (n == 0) return "            "
    k = int(12 * w / n + 0.5); s = ""
    while (length(s) < k) s = s "█"
    while (length(s) < 12) s = s "░"
    return s
}

END {
    for (i = 1; i <= nrooms; i++) {
        r = order[i]
        o = (opp[r] == "" ? "(no match)" : opp[r])
        v = (res[r] == "" ? "?" : res[r])
        walk = (how[r] != "board" && turns[r] == 0)      # never left team preview

        if (v == "?") { unfinished++; if (opp[r] != "") stranded++; continue }

        tot[v]++
        oseen[o] = 1; ow[o, v]++; ogames[o]++
        # per-team is BOARD-ONLY, same rule as per-archetype: a walkover
        # (opponent no-show win, or an orphan clock-loss from the double-match
        # bug) says nothing about the TEAM that happened to sit in the slot —
        # counting them flattered rain's pelippers (+2W) and dinged whichever
        # team drew an orphan (2026-07-30). Walkovers stay in the session
        # summary above, which is about the account, not the teams.
        if (!walk) { tseen[team[r]] = 1; tw[team[r], v]++ }
        if (have_arch) {
            ar = arch_of[team[r]]; if (ar == "") ar = "?"
            aseen[ar] = 1
            if (!walk) apw[ar, v]++          # archetype record is board-only
        }
        fres[src[r], v]++
        endkind[how[r]]++
        if (walk) { walkover[v]++; walk_by[o]++ }
        else {
            pw[v]++                                      # played-out record
            oplay[o, v]++
            if (turns[r] > 0) { tsum[v] += turns[r]; tn[v]++; otsum[o] += turns[r]; otn[o]++ }
        }
        form = form v
    }

    W = tot["W"]; L = tot["L"]; T = tot["T"]; N = W + L + T
    printf "\n"
    if (nfiles == 1) {
        fn = files[1]
        printf "  %s\n", fn
        printf "  %s → %s  ·  %s  ·  %d slots, %d killed by timeout\n",
               substr(t0[fn], 12), substr(t1[fn], 12), dur(t0[fn], t1[fn]),
               slots[fn], killed[fn]
        printf "  config  %s\n", cfg(fn)
    } else {
        printf "  %d sessions  ·  %d slots, %d killed by timeout\n",
               nfiles, sum_slots(), sum_killed()
    }
    printf "\n"

    printf "  all games   %s  %-12s %s  n=%-4d %s\n",
           bar(W, L), rec(W, L, T), rate(W, L), N, wilson(W, L)
    if (walkover["W"] + walkover["L"] + walkover["T"] > 0)
        printf "  board only  %s  %-12s %s  n=%-4d %s\n",
               bar(pw["W"], pw["L"]), rec(pw["W"], pw["L"], pw["T"]),
               rate(pw["W"], pw["L"]), pw["W"] + pw["L"] + pw["T"],
               wilson(pw["W"], pw["L"])
    if (unfinished)
        printf "  %d unfinished (%d had an opponent — in-flight or abandoned)\n",
               unfinished, stranded
    nslots = (nfiles == 1 ? slots[files[1]] : sum_slots())
    if (nslots > nrooms)
        printf "  %d slots never matched (empty queue)\n", nslots - nrooms
    printf "\n"

    # ---- how games ended --------------------------------------------------
    printf "  ended on the board %d  ·  on the clock %d (%d ours, %d theirs, %d both)",
           endkind["board"], endkind["our_clock"] + endkind["opp_clock"] + endkind["both_clock"],
           endkind["our_clock"], endkind["opp_clock"], endkind["both_clock"]
    wv = walkover["W"] + walkover["L"] + walkover["T"]
    if (wv) printf "\n  of which %d never left team preview (walkovers: %dW-%dL-%dT)",
                   wv, walkover["W"] + 0, walkover["L"] + 0, walkover["T"] + 0
    printf "\n\n"

    # ---- per-opponent -----------------------------------------------------
    printf "  per-opponent\n"
    printf "    %-16s %-12s %6s   %-12s %6s  %6s\n",
           "opponent", "record", "rate", "board only", "rate", "turns"
    n = 0
    for (o in oseen) { ord[++n] = o; cnt[o] = ogames[o] }
    for (i = 1; i < n; i++)
        for (j = i + 1; j <= n; j++)
            if (cnt[ord[j]] > cnt[ord[i]]) { s = ord[i]; ord[i] = ord[j]; ord[j] = s }
    for (i = 1; i <= n; i++) {
        o = ord[i]
        printf "    %-16s %-12s %6s   %-12s %6s  %6s\n", o,
               rec(ow[o, "W"], ow[o, "L"], ow[o, "T"]),
               rate(ow[o, "W"], ow[o, "L"]),
               rec(oplay[o, "W"], oplay[o, "L"], oplay[o, "T"]),
               rate(oplay[o, "W"], oplay[o, "L"]),
               (otn[o] ? sprintf("%6.1f", otsum[o] / otn[o]) : "     -")
    }
    printf "\n"

    # ---- game length ------------------------------------------------------
    if (tn["W"] || tn["L"])
        printf "  avg length   wins %.1f turns (n=%d)  ·  losses %.1f turns (n=%d)\n\n",
               (tn["W"] ? tsum["W"] / tn["W"] : 0), tn["W"] + 0,
               (tn["L"] ? tsum["L"] / tn["L"] : 0), tn["L"] + 0

    # ---- per-team ---------------------------------------------------------
    if (show_teams) {
        printf "  per-team  (%d teams, pool rotation, board-only)\n", length(tseen)
        n = 0
        for (t in tseen) { tord[++n] = t; tc[t] = tw[t, "W"] + tw[t, "L"] + tw[t, "T"] }
        for (i = 1; i < n; i++)
            for (j = i + 1; j <= n; j++) {
                a = tord[i]; b = tord[j]
                ra = (tw[a, "W"] + tw[a, "L"]) ? tw[a, "W"] / (tw[a, "W"] + tw[a, "L"]) : -1
                rb = (tw[b, "W"] + tw[b, "L"]) ? tw[b, "W"] / (tw[b, "W"] + tw[b, "L"]) : -1
                if (rb > ra || (rb == ra && tc[b] > tc[a])) { tord[i] = b; tord[j] = a }
            }
        singles_w = singles_l = singles_n = 0
        for (i = 1; i <= n; i++) {
            t = tord[i]
            if (tc[t] < 2) {
                singles_n++; singles_w += tw[t, "W"]; singles_l += tw[t, "L"]
                continue
            }
            printf "    %-34s %-10s %6s\n", t,
                   rec(tw[t, "W"], tw[t, "L"], tw[t, "T"]),
                   rate(tw[t, "W"], tw[t, "L"])
        }
        if (singles_n)
            printf "    %-34s %-10s %6s  (one game each — noise)\n",
                   "+ " singles_n " other team" (singles_n == 1 ? "" : "s"),
                   rec(singles_w, singles_l, 0),
                   rate(singles_w, singles_l)
        printf "\n"
    }

    # ---- per-archetype (pooled across teams, board-only) ------------------
    # per-team n is noise (5-10 games each); pooling by archetype makes the
    # winrate readable. A CI that clears the pool mean is an actual keep/drop
    # signal; overlapping "—" is indistinguishable. Weather groups are solid
    # (ability-definitive); stall/HO/balance are heuristic (see classify_pool.py).
    if (have_arch) {
        pm = (pw["W"] + pw["L"]) ? pw["W"] / (pw["W"] + pw["L"]) : 0
        printf "  per-archetype  (board-only; pool mean %.0f%%, a CI clearing it = keep/drop)\n",
               100 * pm
        printf "    %-14s %4s  %-11s %6s  %-14s  %s\n",
               "archetype", "tms", "record", "rate", "95% CI", "vs pool"
        na = 0
        for (ar in aseen) aord[++na] = ar
        for (i = 1; i < na; i++)
            for (j = i + 1; j <= na; j++) {
                ra = (apw[aord[i],"W"]+apw[aord[i],"L"]) ? apw[aord[i],"W"]/(apw[aord[i],"W"]+apw[aord[i],"L"]) : -1
                rb = (apw[aord[j],"W"]+apw[aord[j],"L"]) ? apw[aord[j],"W"]/(apw[aord[j],"W"]+apw[aord[j],"L"]) : -1
                if (rb > ra) { sw = aord[i]; aord[i] = aord[j]; aord[j] = sw }
            }
        for (i = 1; i <= na; i++) {
            ar = aord[i]
            aWL = apw[ar,"W"] + apw[ar,"L"]
            nteams = 0
            for (t in tseen) { a2 = arch_of[t]; if (a2 == "") a2 = "?"; if (a2 == ar) nteams++ }
            flag = "—"
            if (aWL > 0) {
                p2 = apw[ar,"W"] / aWL
                c2 = (p2 + Z*Z/(2*aWL)) / (1 + Z*Z/aWL)
                h2 = Z * sqrt(p2*(1-p2)/aWL + Z*Z/(4*aWL*aWL)) / (1 + Z*Z/aWL)
                if (c2 - h2 > pm)      flag = "above ✓"
                else if (c2 + h2 < pm) flag = "below ✓"
            }
            printf "    %-14s %4d  %-11s %6s  %-14s  %s\n", ar, nteams,
                   rec(apw[ar,"W"], apw[ar,"L"], apw[ar,"T"]),
                   rate(apw[ar,"W"], apw[ar,"L"]), wilson(apw[ar,"W"], apw[ar,"L"]), flag
        }
        printf "    members (lead mon):\n"
        for (i = 1; i <= na; i++) {
            ar = aord[i]; lst = ""
            for (t in tseen) {
                a2 = arch_of[t]; if (a2 == "") a2 = "?"
                if (a2 == ar) lst = lst (lst == "" ? "" : ", ") leadname(t)
            }
            printf "      %-12s %s\n", ar, lst
        }
        printf "\n"
    }

    # ---- per-session ------------------------------------------------------
    if (nfiles > 1) {
        printf "  per-session\n"
        for (i = 1; i <= nfiles; i++) {
            fn = files[i]
            tag = fn; sub(/^overnight_/, "", tag); sub(/_ladder\.log$/, "", tag)
            printf "    %-18s %-12s %6s  %7s  %s\n", tag,
                   rec(fres[fn, "W"], fres[fn, "L"], fres[fn, "T"]),
                   rate(fres[fn, "W"], fres[fn, "L"]), dur(t0[fn], t1[fn]),
                   cfg(fn)
        }
        printf "\n"
    }

    # ---- form -------------------------------------------------------------
    if (form != "") {
        printf "  form (oldest → newest)\n"
        for (i = 1; i <= length(form); i += 40)
            printf "    %s\n", substr(form, i, 40)
        printf "\n"
    }
}

function sum_slots(   f, s) { for (f in slots) s += slots[f]; return s }
function sum_killed(  f, s) { for (f in killed) s += killed[f]; return s }

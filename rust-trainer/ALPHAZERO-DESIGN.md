# AlphaZero-AI — master design (vaiheet representation → net → trainloop → train)

_Laadittu 2026-06-02. Pohjana luettu koodi: `src/ai/nn/{features,candidates,mlp,policy,search,value,sandbox}.ts`
ja `rust-trainer/crates/cp-ai/src/{features,candidates,mlp,policy,search,value,metrics}.rs`,
`reward.rs`, `golden/SCHEMA.md`, `bin/parity.rs`. Signaalit: [REWARD-DESIGN.md](REWARD-DESIGN.md).
Periaate: **additiivinen** — olemassa oleva GA/parity-polku säilyy byte-identtisenä, AlphaZero rakennetaan rinnalle._

## 0. Mitä jo on (älä rakenna uudelleen)

- **Forward-model:** `cp-sim::Game` on `Clone`, deterministinen, parity-exact (8/8 golden-tracea). Tämä on AlphaZeron kallein edellytys ja se on valmis.
- **Toimintoavaruus on jo candidate-pohjainen:** `enumerate()` tuottaa ehdokkaat (intent × kohderuutu), jokaisella `local`-vektori (LOCAL_DIM=16, joista 6 spatiaalista: enemy/own/neutral-naapurit, dist-own-HQ, dist-nearest-enemy-tile, frontier). Policy rankkaa ehdokkaita → ei tarvita kiinteää grid-policya.
- **Test-time MCTS on jo TS:ssä JA Rustissa:** `search.ts` ↔ `search.rs` (PUCT, priorit netistä, leaf eval static/value/rollout). `sandbox.ts` ajaa headless-pelin snapshotista. `value.ts`/`value.rs` value-verkko. → AlphaZeron **inferenssipolku selaimessa on jo olemassa.**
- **Value-verkko + Adam-treenaaja** Rustissa (`value.rs`, `bin/value_train.rs`).

**Johtopäätös:** AlphaZero ≠ uusi pino tyhjästä. Se on (1) rikkaampi **edustus**, (2) **policy+value-pää** samassa verkossa, (3) **self-play + gradient-treenaaja** joka korvaa GA:n, käyttäen olemassa olevaa MCTS:ää ja forward-modelia. Inferenssi selaimessa = jo rakennettu `search.ts`-polku.

## 1. Edustus (vaihe `representation`)

Säilytä **board-koosta riippumaton** suunnittelu (vaatimus: toimii millä tahansa kartalla). EI täyttä CNN/grid-policya — se rikkoisi invarianssin ja koko parity-pohjan. Sen sijaan rikastetaan kaksi olemassa olevaa tasoa:

### 1A. Per-candidate local (palvelee policya + reward-signaaleja P7/P10/N3)
Lisää `tileSpatial()`-funktioon (TS+Rust) puuttuvat signaalipiirteet:
- `distEnemyHq` — Manhattan-etäisyys lähimpään **vihollisen HQ:hon** (P10: hyökkää lähelle tukikohtaa). Sentinel 99 ennen /20.
- `cutVulnerability` — onko ruutu oman HQ-yhteyden kapeikko (artikulaatiopiste omistusgraafissa) (N3: oman alueen katkaisu). Lasketaan: poistetaanko tämä ruutu → katkeaako oma alue HQ:sta.
- `tileUsefulness` — kohderuudun rakennusarvo (tehdas/farmi/voimala vs tyhjä) normalisoituna (P7). Osin jo `targetValue`.

→ LOCAL_DIM 16 → ~19. **Vaikuttaa parity-dimensioon** → re-export golden + parity 8/8. Olemassa oleva GA-champion menee yhteensopimattomaksi — OK, AlphaZero treenataan tyhjästä.

### 1B. Global spatiaaliset summat (palvelee value-päätä — tämä oli tutkimuksen pullonkaula)
Lisää `globalFeatures()`-vektoriin board-invariantteja spatiaalisia skalaareja (ei gridiä):
- `hqToHqDist` — oman ja lähimmän vihollisen HQ:n etäisyys / karttadiagonaali.
- `ownDispersion` — omien ruutujen hajonta (keskipisteen ympärillä) → "hajallaanko vai tiivis".
- `frontierLength` — oman ja vihollisen rajan pituus / oma ruutumäärä (P5/N1 puolustustarve).
- `ownHqExposure` — vihollisruutujen määrä oman HQ:n säteellä R / R-naapuruston koko (uhka).
- `enemyHqExposure` — omien ruutujen määrä vihollisen HQ:n säteellä (P10 hyökkäysetenemä).
- `ownCutRisk` — oman HQ-yhteyden kapeikkojen määrä / oma ruutumäärä (N3).

→ GLOBAL_DIM 36 → ~42. Sama parity-vaikutus.

**Toteutus:** uusi `cp-ai/src/spatial.rs` (+ `src/ai/nn/spatial.ts`) joka laskee nämä `Game`/ObjectManagerista, kutsutaan `features.rs`/`candidates.rs`:stä. Reuse: `get_neighbour_tiles`, `get_owner`, `get_hq_tile`, `get_coordinate`. BFS/articulation oman omistusgraafin yli HQ-yhteyden kapeikoille.

## 2. Verkko (vaihe `net`)

Yksi runko, kaksi päätä (AlphaZero-tyyli):
- **Syöte per ehdokas:** `[global(~42) ++ intent_onehot(11) ++ local(~19)]` ≈ 72-ulotteinen (kuten nyt, kasvatettu).
- **Runko:** MLP (reuse `mlp.rs`/`mlp.ts`), esim. `[72, 48, 32]` tanh. Iso GA-verkkoon nähden mutta gradient-treenaus jaksaa.
- **Policy-pää:** skalaari per ehdokas → softmax ehdokkaiden yli (= P(intent×kohde)). Tämä on jo `policy.ts`-rakenne, vain pää eriytetään.
- **Value-pää:** erillinen skalaari **globaalista** tilasta (ei per-ehdokas) → tanh ∈ [-1,1] (voiton todennäköisyys). Reuse `value.rs`-arkkitehtuuri mutta kytketään samaan koulutussilmukkaan.

Serialisointi: laajenna `weights.ts`/`emit-weights.ts` kantamaan policy- ja value-painot + uudet dimensiot. Inferenssi: `search.ts` käyttää policya prioreihin ja valuea leaf-evaliin (jo rakennettu — vain uudet painot + dimensiot).

## 3. Koulutussilmukka (vaihe `trainloop`)

Korvaa GA gradient-self-playllä:
1. **Self-play:** aja N peliä, joka siirrossa MCTS (reuse `search.rs`) M simulaatiolla. Tallenna jokaisesta päätöksestä `(syöte-ehdokkaat, MCTS visit-count -jakauma π, pelaaja, lopputulos z)`.
2. **Reward/target:**
   - Value-target = `z` (lopputulos voittajan kannalta) **+ potentiaalipohjainen bootstrap** `γΦ(s')−Φ(s)` apusignaalina (REWARD-DESIGN Φ). Φ vain nopeuttaa; ei muuta optimia (Ng 1999).
   - Policy-target = MCTS:n `π` (visit-count-jakauma) → cross-entropy.
3. **Replay buffer:** rengaspuskuri viimeisistä K peliä; minibatch-otanta.
4. **Gradient-treenaaja:** Adam (reuse `value_train.rs`-Adam), yhteis-loss `L = (z−v)² − Σπ·log(p) + c·‖w‖²`.
5. **Iteraatio:** treenaa → korvaa self-play-verkko → toista. **PFSP**-tyylinen vastustajaotanta (poimi useammin verkkoja joita vastaan häviät) [[neural-ai]]-HoF:n päälle.
6. **Checkpoint joka iteraatio** → `models/registry.jsonl` (`registry.ts add`), benchmark vs hard sidecarina → dashboard.

Uudet telemetria-kentät `SeatTelemetry`:hin (Φ:tä ja diagnostiikkaa varten): `own_soldiers_lost`, `own_tiles_lost_via_cut`, income-trendi Δ, `idle_potential`. Parity-vaikutus tarkistettava (telemetria ei saa muuttaa pelilogiikkaa).

## 4. Koulutus + julkaisu (vaihe `train`)

- Aja silmukka (paikallisesti, monisäikeinen — raskas vaihe, mutta vain offline).
- **Benchmark vs hard** (`bench_hard`) = oraakkeli; vertaa win-ratea, ei raakaa lossia.
- Ylennä paras → `registry.ts promote` → `emit-weights.ts` → `src/ai/nn/weights.ts` → `npm run build`.
- Tierit: hard = paljon simejä + value-leaf + argmax; medium/easy = vähemmän simejä + static-leaf + lämpötila/blunder (jo `tiers.ts`/`search.ts`:ssä).
- Verify: parity 8/8, TS-vs-Rust search-parity, selain-smoke + per-move-latenssi (`bench-mcts.ts`).

## 5. Parity-strategia (älä riko tätä) — TARKENNETTU 2026-06-02

**Päätös:** ÄLÄ muuta shipattujen `features.ts`/`candidates.ts`:n LOCAL/GLOBAL-dimensioita. Syy: net-syöte = GLOBAL+INTENT+LOCAL → dimensiomuutos rikkoo (a) selaimeen deployatun `weights.ts`-championin (kiinteä syötedim) ja (b) pakottaa regeneroimaan golden-tracet — viikoiksi, kunnes uusi verkko on koulutettu. Se jättäisi pelin nn-vastustajat rikki koko ajaksi.

**Sen sijaan (additiivinen):**
- `spatial.{rs,ts}` (VALMIS) ovat itsenäisiä laskimia.
- AlphaZero-verkko saa OMAN syötekoonnin (`az_features` / Rust + TS) joka yhdistää nykyiset globaalit + intent + local + uudet spatiaaliset piirteet **uuteen verkkoon**. Vanha GA/parity/deploy-polku pysyy byte-identtisenä.
- `weights.ts` vaihdetaan **atomisesti vasta deployssa** (vaihe `train`), kun uusi yhteensopiva champion on olemassa → peli ei mene rikki välissä.
- Olemassa oleva parity (8/8) suojaa vanhaa polkua; AlphaZero-syötteelle tehdään OMA TS-vs-Rust-feature-parity-testi (kiinteä Game-tila → samat luvut).
- Telemetria-kentät: vain luku-/laskuripäivityksiä jotka eivät vaikuta päätöksiin → vanha parity säilyy.

→ Eli vaihe `representation` = `spatial.{rs,ts}` (valmis) + uusi `az_features`-koonti (rakennetaan osana `net`-vaihetta, koska se on verkon syöte). Ei riskialtista shipatun polun dimensiomuutosta.

## 6. Milestonet (dashboardin vaiheet)

| Vaihe | Sisältö | Valmis kun |
|---|---|---|
| representation | `spatial.{rs,ts}` + uudet local/global-piirteet + golden + parity | parity 8/8, dimensiot kasvaneet, sanity-feature-testit |
| net | policy+value-pää samassa rungossa, serialisointi, TS-inferenssi | forward-pass TS=Rust parity fixed-syötteellä |
| trainloop | self-play→replay→Adam + Φ-shaping + PFSP + checkpoint→registry | yksi iteraatio ajaa päästä päähän, value/policy-loss laskee |
| train | iso ajo + benchmark + ylennys + emit + tierit | win-rate vs hard ylittää nykyisen 15–20 % bändin, julkaistu selaimeen |

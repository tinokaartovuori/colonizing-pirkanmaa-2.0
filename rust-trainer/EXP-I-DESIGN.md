# Exp I — Spatiaalinen / katkaisu-tietoinen policy (pitkän treenin suunnitelma)

_Laadittu 2026-06-03. Pohjana `GAME-COMPLEXITY-AND-TRAINING.md`: kaikki halvat vivut loppuun käytetty,
ceiling on POLICYn spatiaalinen sokeus. Tämä on suunnitelma sille ainoalle vivulle joka voi murtaa ~33 %:n._

## Tavoite & ydinhypoteesi

Peli ratkeaa **HQ-yhteyden katkaisulla** (ota artikulaatioruutu joka irrottaa vihollisen HQ:sta) ja
**voiman keskityksellä**. Nykyinen policy näkee vain 36 globaalia aggregaattia + 11 intentti-onehotia +
16-ulotteisen ei-spatiaalisen lokaalin → se ei voi nähdä "tämä ruutu katkaisee 40 % vihollisesta".
**Hypoteesi:** kun policy saa per-ehdokas SPATIAALISET piirteet — ennen kaikkea offensiivisen
katkaisuarvon — se oppii valitsemaan voittavat siirrot, ja win-rate nousee yli ~33 %:n.

## Invariantit (EI rikota)

- **Parity 8/8 ja live-peli säilyvät koskemattomina.** Kaikki spatiaali-policy-logiikka eristetään
  **AZ-koulutuspolkuun**. Jaettua `candidates.rs`/`policy.rs`/`local_vec`-polkua (jota parity ja
  `select_index` + shipattu `weights.ts` käyttävät) **EI muuteta** → golden-traceja ei tarvitse re-exportata,
  live-peli pyörii vanhalla 63-ulotteisella verkolla kunnes uusi mestari on valmis ja erikseen deployattu.
- Reward = **puhdas win/loss-outcome** (kaikki shaping todettu haitalliseksi: positio→draw, decisiveness→loss,
  aggressio→huonompi). Ei shapingia.
- Ei laadun heikennystä nopeuden vuoksi (sims/cap ovat laatunuppeja). Jätä ~4 ydintä vapaaksi.

## Edustus — uudet per-ehdokas spatiaaliset piirteet

Lisätään AZ-policyn syötteeseen per-ehdokas spatiaalinen lohko (kohderuutu = `action`-enumista):
`policy_input_spatial = [ global(36) | intent-onehot(11) | local(16) | SPATIAL(k) ]`.

SPATIAL-lohko (k ≈ 6), laskettu ehdokkaan kohderuudulle T:
1. **offensive_cut_value** — *kruunu*. Jos T on vihollisen ruutu: rakenna vihollisen `OwnedGraph` ja palauta
   `cut_fraction_of(T)` = paljonko vihollisen ruuduista irtoaa HQ:sta jos otan T:n (1.0 jos T = vihollisen HQ).
   Neutraalille/omalle T:lle 0. **Tämä tekee voittoehdon näkyväksi.**
2. **dist_to_enemy_hq(T)** normalisoituna — kuinka lähellä tappokohdetta.
3. **is_enemy_hq(T)** — 1 jos T on elävän vihollisen HQ.
4. **own_cut_vulnerability(T)** — puolustus: paljastanko itseni katkaisulle (oman graafin cut_fraction).
5. **target_enemy_neighbors / 4** — montako T:n naapuria on vihollisen (painostuspinta).
6. **target_owner_is_enemy** — 1 jos T on vihollisen (Attack) vs neutraali (Expand).

Kaikki ∈ [0,1] (clamp). Lasketaan vain AZ-polulla; jaettu enumerate ei muutu (laskenta tehdään ehdokkaille
MCTS-priorien / self-play-tallennuksen yhteydessä).

## Arkkitehtuuri & koulutus

- **Policy-arch:** `DEFAULT_ARCH_SPATIAL = [POLICY_INPUT_DIM + k, 32, 16, 1]` (hieman leveämpi kuin 24, koska
  syöte kasvaa ja edustus on rikkaampi). **Cold start** (random init) — dim ≠ shipattu 63 → ei warm-startia
  policylle.
- **Value-net:** warm-start olemassa olevasta 41-ulotteisesta spatiaalisesta value-verkosta (az4); value-leaf MCTS.
- **MCTS:** spatiaalinen policy tuottaa priorit (`policy_input_spatial`), value-leaf arvioi lehdet. sims 96+.
- **Self-play:** puhdas outcome z (win +1 / loss −1 / timeout-tie 0). Tallenna spatiaaliset policy-inputit π:n kanssa.
- **Koko:** treeni + benchmark **14×12** (pelin oletus; AI on huonompi isolla kartalla → harjoittele siellä).
  Harkitse myöhemmin kokosekoitusta (spatiaaliset piirteet ovat kokoinvariantteja: fraktioita & normalisoituja).
- **Held-out eval:** `champ_probe` legitiimi win-rate (hardin itse-konkurssipelit poistettu), erillinen seed-virta.

## Robustius pitkälle ajolle (top-notch)

- Auto-checkpoint joka bench-väli (champion.json + value.json + benchmark-history + log).
- **Resumoitavuus:** `--init-policy`/`--init-value` jatkaa keskeytyneestä; truncate-logiikka kunnossa.
- Rinnakkaisuus rayonilla, `--threads` jättää ~4 ydintä vapaaksi.
- Dashboard osoittaa benchmark-trendin live.
- Pitkä ajo: ~150–250 iter (mutta seuraa E:n oppia — pelkkä pidempi ajo ilman edustusta valuu; nyt edustus on korjattu).

## Toteutusvaiheet (verifioi joka askel)

1. **`spatial::offensive_cut_value(g, attacker, target)`** + **yksikkötestit** (pieni lauta jossa T irrottaa N
   vihollisruutua → feature = N/total; HQ → 1.0; neutraali → 0). **KRUUNU — väärä tässä = hukattu ajo.**
2. **`candidate_spatial_features(g, p, c)`** (k-ulotteinen lohko ehdokkaalle) + testit.
3. **`policy_input_spatial` + `POLICY_INPUT_DIM_SPATIAL` + `DEFAULT_ARCH_SPATIAL`** (uusi policy-moduulin osa).
4. **AZ MCTS-priorit** käyttävät spatiaalista syötettä (search.rs: `--spatial-policy`-haara).
5. **selfplay + controller** tallentavat spatiaaliset inputit; **policy_train** kouluttaa (dim-agnostinen jo).
6. **alphazero.rs `--spatial-policy`**: cold-start spatiaalipolicy, value warm-start, puhdas reward, 14×12.
7. **Parity 8/8 -varmistus** (jaettu polku koskematon) + **smoke-ajo** (5 iter, järkevä output) ENNEN pitkää ajoa.
8. **Pitkä ajo** + champ_probe-arvio. Jos voittaa exp-A (33 %) → portaa spatiaalipiirteet TS:ään + deploy.

## Riskit
- Cold-start spatiaalipolicy voi oppia hitaasti (ei warm-startia). Lievennys: value warm-start + value-leaf
  antaa heti järkevän hakusignaalin; sims riittävä.
- Per-ehdokas enemy-graph-rakennus on O(tiles) × ehdokkaat/vuoro → hidastaa self-playta. Lievennys: rakenna
  enemy-graph kerran/vuoro ja jaa ehdokkaiden kesken (kohteet ovat saman vihollisen ruutuja).
- Jos EI voita exp-A:ta: edustus oli oikea suunta mutta riittämätön → seuraava on hierarkkinen/AlphaZero-MCTS
  syvempi haku tai GNN-policy (tutkimusdokumentin reitit).

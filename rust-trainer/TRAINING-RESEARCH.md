# Selvitys: miten *Colonizing Pirkanmaa* -AI tulisi kouluttaa

_Laadittu 2026-06-02. Pohjana: (a) nykyarkkitehtuurin kartoitus (`src/ai/nn/*`, `rust-trainer/`),
(b) deep-research-haku tieteellisestä kirjallisuudesta (25 primäärilähdettä, 23 adversariaalisesti
vahvistettua väitettä). Lähteet listattu lopussa._

---

## 0. Johtopäätös ensin (TL;DR)

**Diagnoosi vahvistui kirjallisuudesta:** ongelma on **edustus- ja toimintoavaruuskatto, ei reward-viritys.**
Verkko vain rankkaa 11 käsin koodattua makro-intentiä, ja kohdevalinta (mihin ruutuun laajennetaan / ketä
hyökätään) on heuristiikkaa. Kaksi riippumatonta peliä-AI-tutkimusta osoittaa suoraan, että **käsin tehdyt
toimintoabstraktiot rajaavat strategia-avaruuden** ja että abstraktion *oppiminen* / laajentaminen voittaa
kiinteät valikot (Moraes AAAI'19; Xu AIIDE'19). Lisää reward-säätöä ei kannata tehdä sokkona.

**Kaksi järkevää reittiä, prioriteettijärjestyksessä:**

| Reitti | Mitä | Kattopotentiaali | Työmäärä / riski |
|---|---|---|---|
| **A. Korjaa edustus nykyisen GA-harnessin sisällä** | Anna verkon (1) **valita kohteet itse** (pointer/per-ruutu-pisteytys ehdokasruuduille) ja (2) nähdä **spatiaalista/entity-tietoa**; vaihda HoF-vastustajan poiminta **PFSP**:ksi; aja **novelty search** -diagnoosi | Keskisuuri–suuri | Pieni–keskisuuri (säilyttää parity-pohjan ja GA:n) |
| **B. AlphaZero/MuZero-tyylinen MCTS-self-play** | Hyödynnä nopeaa, deterministista, parity-varmennettua forward-modelia suoraan suunnitteluun | **Suurin** (superhuman Go/chess/shogi/Atari) | Suurempi (uusi koulutuspino, value/policy-päät) |

**Rust-simun arvio (osa 3):** ympäristönä **erinomainen** koulutukseen (nopea, deterministinen, parity-exact,
forward-model käytettävissä → ihanteellinen MCTS:lle). **Pullonkaula on rajapinta** jonka simu tarjoaa oppijalle:
57-ulotteinen aggregoitu syöte + 11 kiinteää intentiä + heuristinen kohdevalinta. GA:n hyperparametrit
(pop 48 / games 24 / mutaatio-only) ovat kohtuulliset mutta toissijaiset — varsinainen korjaus on havainto/toiminto-rajapinta.

---

## 1. Mitä tiede sanoo (kirjallisuuskatsaus)

### 1A. Koulutusparadigmat: RL vs. neuroevoluutio vs. hybridit

- **Mallipohjainen MCTS + self-play (AlphaZero/MuZero) on vahvin tunnettu resepti** lautapelin­kaltaisille,
  täydellisen informaation peleille. MuZero saavutti superhuman-tason Go/shakki/shogi/Atari yhdistämällä
  MCTS-haun opittuun malliin, joka ennustaa rewardin, policyn ja arvon — **jopa ilman annettuja sääntöjä**
  (Schrittwieser et al., *Nature* 2020). Olennaista meille: AlphaZero käyttää **annettua forward-modelia**
  hakuun — ja meillä *on* sellainen (Rust-simu).
- **Gradientittomat GA/ES skaalautuvat yli pikku-MLP:n** ja ovat seinäkello-ajassa kilpailukykyisiä
  rinnakkaisuudella. "Deep GA" koulutti 4M+ parametrin verkkoja ja oli **nopeampi kuin ES, A3C ja DQN**
  Atari-peleissä (Such et al. 2017). OpenAI ES skaalautui 1000+ työntekijälle (Salimans et al. 2017). **Eli
  spatiaalisen syötteen ja isomman verkon lisääminen nykyisen GA:n alle on mahdollista** — hinta on
  näytetehokkuus (Majid et al. 2021 varoittaa ES:n vaikeudesta hyvin korkeaulotteisessa optimoinnissa).
- **Kumottu (älä oleta):** väite "hybridi-DRL+ES voittaa empiirisesti kumman tahansa yksinään ja on tuottanut
  erittäin vahvaa peliä StarCraftissa" **kaatui verifioinnissa 0–3**. Hybridi ei siis ole todistettu
  oikotie — älä rakenna strategiaa sen varaan.

> **Sovellus:** Deterministinen forward-model tekee AlphaZero/MuZero-reitistä poikkeuksellisen sopivan
> (useimmilla indie-projekteilla ei ole tarkkaa mallia). GA on validi tapa skaalata verkkoa, mutta yksin se ei
> poista edustuskattoa — verkon *syöte ja toiminnot* on korjattava joka tapauksessa.

### 1B. Tila- ja toimintoesitys — **tämä on ydin**

- **Käsin tehdyt makro-toimintovalikot rajoittavat suoraan suorituskykyä.** Moraes et al. (AAAI 2019):
  käsin kirjoitetut strategiajoukot rajaavat käyttäytymistä, ja **abstraktion evoluutio voittaa SoTA-suunnittelijat**.
  Xu et al. (AIIDE 2019): ennalta määrätyt makrosäännöt eivät skaalaudu. → **Suora vahvistus tämän pelin
  diagnoosille.**
- **Vahvat agentit eivät rankkaa kiinteää valikkoa** — ne käyttävät **spatiaalista/entity-syötettä +
  autoregressiivisia pointer-päitä**, jotka faktoroivat toiminnon osiin (tyyppi → yksiköt → kohde → ajoitus).
  AlphaStarissa tämä tuottaa ~10²⁶ toiminnon avaruuden: self-attention yksiköiden yli, scatter-yhteydet,
  syvä LSTM, rekursiivinen pointer-verkko (Vinyals et al., *Nature* 2019).
- **Kevyempi vaihtoehto vaihtelevankokoiselle laudalle:** **graafineuroverkko + autoregressiivinen
  policy-dekompositio** poistaa kiinteän pituuden rajoituksen ja yleistää **zero-shot** eri kokoihin.
  Janisch et al. (ICML 2021): 5-palikan koulutuksella ratkaisee 78 % 20-palikan tehtävistä. Tämä on
  juuri "objektikeskeinen toiminto" -malli, jota expand/attack-ruutuvalinta vaatii.
- **Hierarkkinen RL korjaa juuri meidän rakenteemme.** FeUdal Networks (Vezhnevets et al., ICML 2017):
  **Manager** asettaa tavoitteita matalalla aikaresoluutiolla, **Worker** toimii per tick — tämä **mallintaa
  suoraan "intent vs. kohde" -jaon** ja **korjaa pitkän horisontin credit-assignmentin** (juuri se, mistä
  timeoutit ja tasainen fitness kielivät).

> **Sovellus:** Tämä on selvin pullonkaula. Verkon pitää (1) valita kohteet, ei vain intent, ja (2) nähdä
> ruudukon rakenne. "Intent → kohde" on luonteva autoregressiivinen/feudaalinen jako.

### 1C. Self-play -metodologia

- **Prioritized Fictitious Self-Play (PFSP) + liigakoulutus** tuottaa robustia peliä ja **välttää syklit ja
  unohtamisen** (AlphaStar, *Nature* 2019): kolme roolia — *main agents* (PFSP, painota vastustajia joita
  häviät), *main exploiters* (etsi pääagentin heikkoudet), *league exploiters* (etsi koko liigan heikkoudet);
  säilytä osa puhdasta self-playta.
- **PSRO** (Lanctot et al. 2017) yleistää fiktiivisen self-playn ja kaksoisoraakkelin: pidä politiikkapopulaatio,
  laske empiirinen meta-peli, laske paras vaste sekoitettua vastustajaa vastaan.
- **Autocurricula** (Baker et al. 2019): self-play synnyttää itsestään kasvavan vaikeusasteen.

> **Sovellus:** Nykyinen "uniform sample populaatiosta ∪ HoF" on naiivi. **PFSP-painotus (poimi useammin
> vastustajia joita vastaan häviät) on halpa, korkean ROI:n parannus**, joka istuu suoraan olemassa olevaan
> HoF-koneistoon. Pidä kova heuristiikka edelleen held-out-benchmarkina.

### 1D. Reward & exploration pitkän horisontin peleissä

- **Potentiaalipohjainen reward shaping säilyttää optimaalisen politiikan** (Ng, Harada, Russell 1999) — eli
  tiheän signaalin voi lisätä **muuttamatta sitä mikä on optimaalista**, jos shaping on muotoa γΦ(s')−Φ(s).
  Nykyinen dense-reward EI ole potentiaalimuotoinen → se voi vinouttaa optimia (selittää osan v1/v2/v3-tuloksista).
- **Tasainen fitness voi tarkoittaa exploration-epäonnistumista** harvalla pitkän horisontin rewardilla — ja
  **novelty search** korjaa sen *kun syy on exploration eikä edustus* (Such et al. 2017: GA+novelty ratkaisee
  tehtävän jossa DQN/A3C/ES/GA epäonnistuvat). **Tämä antaa ratkaisevan diagnoosin** (ks. osa 4).
- **Intrinsic motivation** (RND, Burda et al. 2018) on gradienttipohjainen exploration-bonus harvoille
  rewardeille — relevantti vasta jos siirrytään RL:ään.

> **Sovellus:** Ennen lisää reward-virittämistä: (1) tee dense-termeistä potentiaalimuotoiset, (2) aja
> novelty-search-koe erottamaan "exploration-jumi" vs. "edustuskatto".

### 1E. Käytännön ROI ja budjetit pienelle projektille

- GA/ES ovat **massiivisesti rinnakkaistettavissa** ja sopivat nopeaan headless-simuun (Salimans et al. 2017;
  OpenAI ES -blogi). 280 peliä/s / 20 ydintä on hyvä lähtökohta.
- MuZero-luokan tulokset vaativat ison laskennan, mutta **pienen pelin** (pieni lauta, ~11 intentiä,
  deterministinen) AlphaZero-toteutus on indie-mittakaavassa tehty monta kertaa — forward-model on se kallis
  pala, ja se on jo olemassa.

---

## 2. Soveltaminen tähän peliin (konkreettinen tiekartta)

Nykytila kartoitettuna (`src/ai/nn/`): MLP **[57,24,16,1]**, ~1809 param; syöte = **36-ulotteinen aggregoitu
globaali tila** + 11-intent-onehot + 10-ulotteinen ehdokas-piirre; verkko **rankkaa 11 intentiä**; Expand/Attack
valitsevat kohteen **käsin koodatulla heuristiikalla** (`candidates.ts`); turvaskaffold
(`ensureWoodIncome`/`staffIncome`) hoitaa pakolliset; GA mutaatio-only σ 0.18→0.05, pop 48, elite 8, 24 peliä/genomi,
self-play populaatio ∪ HoF.

### Reitti A — korjaa edustus, säilytä GA (suositeltu ensiaskel)

1. **Opittu kohdevalinta (tärkein).** Älä anna heuristiikan valita expand/attack-ruutua. Pisteytä verkolla
   *jokainen ehdokasruutu* (per-ruutu-piirteet) ja valitse argmax — eli "intent" ja "kohde" molemmat verkolta,
   autoregressiivisesti (vrt. Janisch ICML'21, AlphaStar pointer). Tämä laajentaa toimintoavaruuden
   {11 intentiä} → {intent × kohderuutu}. **Muutos koskettaa `candidates.ts` enumerate()/localVec ja TS+Rust
   yhtä aikaa → re-export golden + parity uudelleen.**
2. **Spatiaalinen/entity-syöte.** Lisää per-ruutu- tai naapuruus-piirteitä (oma/vihollinen/neutraali, rakennus,
   uhka) joko pienenä CNN:nä ruudukon yli tai GNN:nä omistus-/naapurigraafissa. Tämä nostaa 57-ulotteisen
   aggregaatin yli — verkko näkee "missä" eikä vain "kuinka monta".
3. **PFSP-vastustajapoiminta.** Korvaa uniform-poiminta populaatio∪HoF:sta painotuksella, joka suosii
   vastustajia joita vastaan genomi häviää (AlphaStar). Halpa, istuu olemassa olevaan HoF:iin.
4. **Potentiaalipohjainen shaping.** Muotoile dense-termit (`reward.rs`) muotoon γΦ(s')−Φ(s) → tiheä signaali
   ilman optimin vinoutusta (Ng 1999).
5. **Harkitse CMA-ES:ää** mutaatio-only-GA:n tilalle pienelle param-määrälle (kovarianssin adaptaatio
   nopeuttaa), mutta tämä on toissijainen verrattuna kohtiin 1–2.

### Reitti B — AlphaZero/MuZero MCTS-self-play (suurin katto)

- **Rust-simu = AlphaZeron "säännöt".** Lisää (policy, value) -päät, aja **MCTS** forward-modelilla, kouluta
  self-play-peleistä. Deterministinen + nopea + parity-exact tekee tästä epätavallisen toteutuskelpoisen.
- Vaatii uuden koulutuspinon (puuhaku, replay, gradienttikoulutus) — isompi investointi, mutta ainoa reitti
  joka on todistetusti tuottanut superhuman-strategiapeliä annetulla mallilla.
- Voi yhdistää reitin A edustuskorjauksiin (spatiaalinen syöte hyödyttää myös policy/value-päitä).

### Mihin kukin kartoittuu (kysymäsi mappays)

| Suositus | Pikku-MLP (rankkaa intentit) | Forward-model-simu | GA-harness |
|---|---|---|---|
| Opittu kohdevalinta (A1) | **Korvaa** kiinteän intent-rankkauksen intent×kohde-faktoroinnilla | — | Säilyy (fitness ennallaan) |
| Spatiaalinen/GNN-syöte (A2) | Kasvattaa verkkoa; GA skaalaa (Such'17) | — | Säilyy, näytetehokkuus laskee |
| PFSP (A3) | — | — | **Korvaa** uniform-poiminnan HoF:ssa |
| Potentiaali-shaping (A4) | — | — | `reward.rs`-muutos |
| Novelty-search-diagnoosi (osa 4) | — | — | Lisää käyttäytymis-deskriptorin |
| MuZero/AlphaZero (B) | Korvaa policy/value-päillä | **Hyödyntää suoraan** (MCTS) | Korvautuu gradientti-self-playllä |

---

## 3. Ovatko Rust-simun parametrit hyvät koulutukseen?

Erotetaan kaksi tasoa, koska vastaus on erilainen:

### 3.1 Simu *koulutusympäristönä* → **erinomainen, säilytä**
- **Nopea** (~280 peliä/s / 20 ydintä), **deterministinen**, **bit-for-bit parity** oikean pelin kanssa
  (8/8 golden-tracea, 2052 päätöstä + 4338 sormenjälkeä — varmistettu eilen). Tämä on **suuri etu**, jota
  useimmilla projekteilla ei ole.
- **Forward-model on käytettävissä** → mahdollistaa MCTS/suunnittelun (reitti B). Tämä on AlphaZeron kallein
  edellytys, ja se on jo valmiina.
- → Älä koske tähän. Se on koulutuksen vahvuus, ei heikkous.

### 3.2 Simun *oppijalle tarjoama rajapinta* → **tässä on katto**
- **Havainto:** 36-ulotteinen aggregoitu globaali tila on **board-koosta riippumaton mutta sokea sijainnille**.
  Verkko ei voi oppia "puolusta luoteisrajaa" tai "laajene tähän suuntaan". → korjaa (A2).
- **Toiminnot:** 11 kiinteää intentiä + **heuristinen kohdevalinta**. Tämä on kirjallisuuden tunnistama
  kattomekanismi (Moraes AAAI'19, Xu AIIDE'19). → korjaa (A1).
- **Turvaskaffold** takaa ettei AI mene konkurssiin, mutta samalla **syö osan toimintobudjetista** ja kaventaa
  opittavaa tilaa — tämä kannattaa tarkistaa championin trace-ajossa (handoffin "DO THIS FIRST").

### 3.3 GA-hyperparametrit → **kohtuulliset, toissijaiset**
- **Mutaatio-only / ei crossoveria:** perusteltu (NN-permutaatio-ongelma; Deep GA toimii ilman crossoveria,
  Such'17). OK.
- **Pop 48 / elite 8 / 24 peliä/genomi:** pieni mutta toimiva. Jos siirryt isompaan verkkoon (A2), näytetehokkuus
  laskee → kasvata populaatiota/pelejä tai harkitse CMA-ES/OpenAI-ES-rinnakkaistusta.
- **σ-anneal 0.18→0.05, self-play populaatio∪HoF:** kunnossa; **ainoa selvä parannus on PFSP-poiminta (A3)**.
- **Reward:** dense-termit eivät ole potentiaalimuotoisia → mahdollinen optimin vinoutus (A4). Mutta **reward ei
  ole pääongelma** — se on edustus.

**Tiivis vastaus:** Simun *moottoriparametrit* (nopeus, determinismi, forward-model, parity) ovat erinomaiset
koulutukseen. Simun *oppijalle altistama havainto/toiminto-rajapinta* on se mikä rajoittaa — ja GA:n
hyperparametrit ovat kunnossa mutta eivät ratkaise mitään ennen kuin rajapinta korjataan.

---

## 4. Ratkaiseva seuraava koe (halpa, erottaa hypoteesit)

Ennen isoa työtä — **aja novelty-search-variantti nykyisellä GA:lla** (käyttäytymis-deskriptori esim.
[lopullinen tile-frac, hyökkäysten määrä, rakennusjakauma]):
- **Jos novelty irrottaa jumista** (fitness/tile-frac nousee) → syy oli **exploration**, ei edustus → halvempi korjaus.
- **Jos ei** → vahvistaa **edustuskaton** → siirry suoraan reittiin A1/A2.

Tämä yhdistettynä handoffin championin **trace-ajoon** (valitseeko se koskaan Expand/Attack; generoidaanko
Expand-ehdokkaita; syökö skaffold budjetin) antaa varman suunnan ennen isoa investointia.

---

## Lähteet (kaikki primäärilähteitä, adversariaalisesti verifioitu 3-äänin)

1. **Moraes et al., AAAI 2019** — käsin tehdyt strategiajoukot rajaavat; abstraktion evoluutio voittaa SoTA-suunnittelijat. `ojs.aaai.org/index.php/AAAI/article/download/4072/3950`
2. **Xu et al., AIIDE 2019** — ennalta määrätyt makrosäännöt eivät skaalaudu. arXiv:1812.00336
3. **Vinyals et al. (AlphaStar), *Nature* 2019** — spatiaalinen/entity-syöte, autoregressiiviset pointer-päät, PFSP-liigakoulutus. DeepMind PDF
4. **Janisch, Pevný, Lisý, ICML 2021** — GNN + autoregressiivinen policy-dekompositio, zero-shot-yleistys. arXiv:2009.12462
5. **Vezhnevets et al. (FeUdal Networks), ICML 2017** — manager/worker-hierarkia, pitkän horisontin credit-assignment. arXiv:1703.01161
6. **Schrittwieser et al. (MuZero), *Nature* 2020** — MCTS + opittu malli, superhuman. arXiv:1911.08265
7. **Such et al. 2017 (Deep GA / novelty)** — GA skaalaa 4M+ param, novelty irrottaa exploration-jumin. arXiv:1712.06567
8. **Salimans et al. 2017 (OpenAI ES)** — rinnakkaistus 1000+ työntekijää. arXiv:1703.03864
9. **Majid et al. 2021** — evoluutio-RL-katsaus; ES:n vaikeus korkeaulotteisessa optimoinnissa. arXiv:2110.01411
10. **Ng, Harada, Russell 1999** — potentiaalipohjainen reward shaping säilyttää optimin. andrewng.org
11. **Burda et al. 2018 (RND)** — intrinsic exploration. arXiv:1810.12894
12. **Lanctot et al. 2017 (PSRO)** — politiikkapopulaatio + meta-peli. arXiv:1711.00832
13. **Baker et al. 2019** — autocurricula self-playssa. arXiv:1909.07528

**Kumotut väitteet (EI suositella):** "hybridi-DRL+ES voittaa kumman tahansa yksinään / käytetty StarCraftissa"
(kaatui 0–3); "ES-skaalauksen näytetehokkuus-argumentti yksiselitteinen" (1–2, epävarma).

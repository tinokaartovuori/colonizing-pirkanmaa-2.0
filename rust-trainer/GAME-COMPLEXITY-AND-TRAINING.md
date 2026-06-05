# Pelin kompleksisuus ja mitä se vaatii AI:n koulutukselta

_Laadittu 2026-06-03. Pohjana: 5 rinnakkaista mekaniikka-auditia (win/lose + HQ-katkaisu,
taistelu, talous, kartta/maasto, AI:n toimintoavaruus) + empiiriset mittaukset (champ_probe
per-outcome + intent-histogrammi, karttakokopyyhkäisy, hard-vs-hard pelipituus). Tämä dokumentti
selittää **mekaanisesti** miksi AI on jumissa ~33 %:ssa ja mitä koulutuksen pitää korjata._

---

## 0. TL;DR

Peli on **pohjimmiltaan spatiaalinen** kahdella tavalla, joita nykyinen AI ei näe eikä osaa:

1. **Voitto = HQ-yhteyden katkaisu (graafiongelma).** Joka vuoron lopussa lasketaan jokaiselle
   pelaajalle 4-suuntainen BFS-vuototäyttö HQ:sta. Jos vihollisen alue katkeaa HQ:sta, se osa
   **menee neutraaliksi (yksiköt tuhoutuvat)**; jos HQ vallataan, **koko vihollisen alue siirtyy
   valloittajalle**. Voittava siirto on usein yhden "silta-ruudun" (artikulaatiopiste) ottaminen.
   Tämä on graafi-leikkaus-päättelyä, jota **aggregoitu piirrevektori ei voi esittää.**
2. **Hyökkäys = voiman keskitys (kynnysmekaniikka).** Valtaus onnistuu vain jos
   `hyökkääjät > puolustajat` **yhdellä ruudulla yhdellä vuorolla** (deterministinen, ei sattumaa).
   Ruudulla on max 3 yksikköä → 3 sotilaan ruutu on murtumaton. Palasittainen hyökkäys (1 sotilas
   2 puolustajaa vastaan) **menettää kaikki hyökkääjät joka vuoro.**

Nykyinen AI epäonnistuu **kahdesta rakenteellisesta syystä**, jotka nyt on mekaanisesti todistettu:

- **Ongelma 1 — armeija jää rakentamatta (cap-1-ansa).** Sotilaskatto = HQ(+1) + Outpost(+3).
  AI rakentaa Outpostin **0.1 % päätöksistä** ja Minen 1.2 % → jää ~1 sotilaan kattoon → voi vallata
  vain **puolustamattomia** ruutuja. Voitot tulevat vain kun vastustaja romahtaa.
- **Ongelma 2 — katkaisu-voittoehto on näkymätön.** Kohdevalinta käyttää maastoarvoa + "HQ ensin",
  **ei mitään signaalia siitä että ruutu katkaisisi vihollisen HQ:n.** Vaikka armeija olisi, AI ei
  tietäisi mihin iskeä.

**Hard-botti (benchmark) HARDKOODAA juuri sen mitä NN ei opi:** 5 outpostia, 7 sotilaan
iskuvoima, Mine→Outpost-ketju, 7 hyökkäystä/vuoro. Siksi hard on koherentti armeijanrakentaja ja
NN on talouskyhjääjä — ja siksi olemme jumissa ~33 %:ssa.

---

## 1. Pelin mekaniikka (auditin tulokset)

### 1A. Voitto/häviö ja turn-flow
- **Voitto** on emergentti: viimeinen jäljellä `player_order`:ssa voittaa. Kaksi reittiä poistaa muut:
  (a) **dominanssi ≥70 % ruuduista** → kaikki muut hävitään + neutralisoidaan; (b) **0 omaa objektia**
  (HQ + tilet menetetty) → häviö. (`managers.rs:969-1007`)
- **Piilotettu häviöehto:** mikä tahansa **negatiivinen resurssi** → välitön häviö + neutralisointi
  (`managers.rs:978-987`). Talous on siis myös tappouhka, ei vain kasvumittari.
- **Vuorojärjestys (end_turn):** tuotanto → palkat → valtaukset (kaikki ruudut) → **HQ-yhteyden
  katkaisu (kaikki viholliset)** → voitto/häviötarkistus → vuoronvaihto. Järjestys on ratkaiseva:
  yksi onnistunut isku voi samalla vuorolla katkaista ja romauttaa vihollisen kokonaan.

### 1B. HQ-yhteyden katkaisu — strateginen ydin
- `get_hq_connected_tiles`: 4-suuntainen BFS HQ:sta omia ruutuja pitkin. **Diagonaalit eivät yhdistä.**
  Valloitettu HQ ⇒ ei juurta ⇒ ei yhtään yhdistettyä ruutua.
- Per vihollinen, joka vuoro: ruudut jotka eivät yhdisty HQ:hon → joko **konfiskoidaan** (jos
  vastustajalla ei ole HQ:ta) tai **menevät neutraaliksi** (yksiköt tuhotaan, rakennukset jäävät).
- **Seuraus:** ota se yksi ruutu joka yhdistää vihollisen HQ:n ulompaan klusteriin → klusteri irtoaa.
  Ota HQ → kaikki vihollisen ruudut sinulle. Tämä on **artikulaatiopiste-/min-cut-päättelyä.**

### 1C. Taistelu — voiman keskitys
- Valtaus: `omat_sotilaat > vihollis_sotilaat` **yhdellä ruudulla**, ei Outpostia tilellä. Deterministinen.
- **Voitto:** puolustajat tuhoutuvat, hyökkääjät säilyvät (0 tappiota). **Häviö/tasapeli:** kaikki
  hyökkääjät tuhoutuvat. Outpost = **murtumaton**. Ruudun kapasiteetti 3 → 3 sotilasta = linnoitus.
- Sotilas liikkuu 1 ruudun/vuoro, jokin oman alueen vierestä. Massaaminen vaatii monivuoroista
  ennakkoasemointia.

### 1D. Talous — pakottaa valinnan laajennus vs. armeija
- Alku: **raha 400, puu 200, kivi 100, metalli 25**.
- **Sotilas: 200 raha + 50 metalli, palkka 30/vuoro.** Metalli **vain Mineistä**. Sotilaskatto:
  **HQ +1, Outpost +3** (Outpost 650 raha + 300 metalli + jatkuva −50 raha/−15 metalli).
- Tuore pelaaja voi pitää **vain 1 sotilaan** kunnes rakentaa Outpostin. → Armeija vaatii ketjun
  **Mine (metalli) → Outpost (paikat) → Soldiers**, joka kilpailee laajennuksen kanssa samasta
  metallista, yksikköpaikoista ja ylläpidosta. Tämä on **aito, syvä strateginen jännite.**
- Farmi: ~38.75 nettorahaa/vuoro/työläinen (175 per 4 vuoroa, ei skaalaudu lisätyöläisillä).

### 1E. Kartta/maasto — luo spatiaalisen rakenteen
- 5 maastotyyppiä. **Yksi mutkitteleva joki** = liike- ja yhteyseste, ylitettävissä vain **sillalla**
  (suoralla jokiruudulla); mutkajoki **halkaisee maan pysyvästi.** Vuoret läpäistäviä (vain Mine).
- Maasto **hardgateaa talouden** (ei vuorta → ei metallia → ei armeijaa). Kartat satunnaisia ja
  epäsymmetrisiä, **korkea seed-varianssi** → strateginen tilanne arvotaan joka peli uudelleen.
- → Maasto luo juuri ne kapeikot ja siltaruudut joiden ympärillä katkaisumekaniikka pyörii.

### 1F. AI:n toimintoavaruus — mikä on ilmaistavissa, mikä ei
- **Attack-primitiivi ON ekspressiivinen:** yksi kandidaatti voi massata `puolustajat+1` sotilasta
  yhdelle ruudulle yhdellä vuorolla (feasibility-portti, ei tihkuttamista). Mine→Outpost→Soldier-ketju
  on myös ilmaistavissa kandidaatteina.
- **MUTTA:**
  - **Ei katkaisu-tietoisuutta:** Expand-kohteet = pelkkä maastoarvo; Attack = "HQ ensin" mutta
    ei signaalia kapeikon/sillan katkaisuarvosta. Voittoehto on näkymätön kohdevalinnalle.
  - **Talouspainotteinen safety-scaffold** ajaa joka vuoro ENNEN ja JÄLKEEN jokaisen opitun siirron;
    täyttää työläiset/talouden, **ei koskaan armeijaa.**
  - **Cap-1-ansa:** HireSoldier näkyy vain kun sotilaspaikkaa on vapaana; HQ-only (cap 1) → 1 sotilaan
    jälkeen katoaa. BuildOutpost on litteä `net_delta −50` -kandidaatti **ilman signaalia että se
    nostaa kattoa**; HireSoldierin `soldier_cap_gain = 0`. → policy on rakenteellisesti ohjattu
    pysymään cap-1:ssä.

---

## 2. Empiirinen todiste (mittaukset)

### 2A. champ_probe — käyttäytyminen lopputuloksittain (az8, 200 peliä vs hard)
| Lopputulos | osuus | champ tile% | hard tile% | champ sot. | hard sot. | champ rak. |
|---|---|---|---|---|---|---|
| VOITTO | ~30 % | 38–46 % | **0 %** | **~1.5** | 0 | ~17–20 |
| HÄVIÖ | ~30 % | 0 % | 60–65 % | 0 | ~2.5 | 0 |
| TIMEOUT | ~38 % | 13–22 % | 32–40 % | ~0.3 | ~1.5 | 8–12 |
- Voitot ovat **oikeita valloituksia**, mutta vain ~1.5 sotilaalla ja vain kun vihollinen on
  romahtanut (0 sotilasta/rakennusta). Timeouteissa AI on jäljessä joka akselilla.

### 2B. Intent-histogrammi — AI YRITTÄÄ taistella mutta ei rakenna armeijaa
- Valitut (az8): Pass 55 %, Expand 17 %, **HireSoldier 10.5 %, Attack 9.3 %**, BuildFarm 5 %,
  **BuildMine 1.2 %, BuildOutpost 0.1 %, BuildNuclear 0 %.**
- Kun tarjolla, Attack valitaan 69 %, Expand 65 % → **se haluaa hyökätä, mutta cap-1 → ~1.5 sotilasta
  → palasittaiset iskut → häviää puolustetut ruudut.** BuildOutpost ~0 % = cap-1-ansa empiirisesti.

### 2C. Reward-virityksen umpikuja (3 koetta)
- Positio-Φ (B) → draw-happy. Decisiveness (F) → loss-spike, päätyi 17.5 % win / 60 % timeout.
  Aggressio-Φ (az8) → ~28.5 %, **baselineä (exp-A 33.5 %) HUONOMPI.** → **Reward ei ole vipu.**

### 2D. Karttakokopyyhkäisy (exp-A vs hard, static leaf, 120 peliä)
| Koko | WIN | LOSS | TIMEOUT | voittajan tile% |
|---|---|---|---|---|
| 12×12 | **34.2 %** | 44.2 % | 21.7 % | 47.8 % |
| **14×12 (pelin oletus)** | 26.7 % | 43.3 % | 30.0 % | 44.6 % |
| 18×13 | 28.3 % | 45.0 % | 26.7 % | 37.3 % |
| 22×14 | **20.0 %** | 43.3 % | 36.7 % | 27.3 % |
- **AI on selvästi HUONOMPI isommalla kartalla** (34 %→20 %), timeoutit nousevat (22 %→37 %), ja voittojen
  tile% laskee → isolla kartalla se ei ehdi laajentua/tavoittaa vihollista. **33 % on optimistinen oikealle
  pelille; 14×12:lla ~30 %, 22×14:llä vain 20 %.** Iso kartta = laajennus & etäisyyden kurominen ratkaisee,
  juuri kuten epäilit.

### 2E. MCTS-syvyys (sims) ei auta (exp-A, 12×12, 120 peliä)
| sims | WIN |
|---|---|
| 48 | 35.8 % |
| 96 | 34.2 % |
| 200 | 33.3 % |
- **Lisää hakua ei nosta win-ratea** (jopa hieman laskee) → vahvistaa: kyse on edustuskatosta, ei haun syvyydestä.

### 2F. Benchmark-integriteetti — vääristääkö hardin oma konkurssi mittausta?
Tarkistettu, koska jos hard tekee itsemurhan taloudella, "voittomme" eivät ole meidän ansiotamme.
- **Solo (ei vastustajaa): hard menee konkurssiin 2.3 % (12×12) / 3.3 % (14×12) peleistä**, marginaali vain −15.
  Eli se VOI itse kaatua talouteen ilman ulkoista syytä (luultavasti oma sotilas-/outpost-ylikulutus).
- **Kilpailullisesti (hard vs hard): 0/52 konkurssia terveenä** — kaikki tapahtuivat ~0 tilellä (jo leikattu).
- **Oikeassa benchmarkissa (champ vs hard): 0/200 konkurssia terveenä** molemmilla koolla → **vääristymä = 0.**
- → Paineen alla hard ei ylikuluta; itsemurha tapahtuu vain matalan paineen sooloilussa. `champ_probe` raportoi
  nyt **legitiimin win-raten** (terveenä-konkurssiin-menneet pelit poistettu) pysyvänä varmistuksena.
- **exp-A rehellinen baseline: 33.5 % (12×12), 30.0 % (14×12)** — legitiimi = raaka.

---

### 2G. Ratkaiseva koe — pakotettu armeija HUONONTAA (champ_probe --force-military, exp-A, 14×12)
| | WIN | LOSS | TIMEOUT |
|---|---|---|---|
| baseline | **30.0 %** | 43.5 % | 26.5 % |
| + pakotettu armeija (Mine→Outpost→sotilaat joka vuoro) | **25.5 %** | **56.5 %** | 18.0 % |
- Ilmaisen armeijan antaminen **laski voittoa (30→25.5 %) ja nosti häviöt (43→56 %).** Sotilas-/outpost-ylläpito
  kuivattaa talouden, eikä pelkkä sotilaiden OLEMASSAOLO auta koska NN ei osaa **koordinoida** voimaa (massata,
  ajaa vihollisen HQ:lle, tehdä katkaisua). → **Sitova rajoite on ② (spatiaalinen koordinointi), EI ① (armeijan
  rakentaminen).** Exp G (vs-hard) + tämä koe yhdessä sulkevat ①-reitin: ongelma on voiman KÄYTTÖ, ei sen puute.

### 2H. Exp G tulos — vs-hard-frac ei muuttanut käytöstä
- az9 (vs-hard-frac 0.5, 14×12): legitiimi win **19.4 %** (alle 30 % baselinen), BuildOutpost yhä 0.2 %,
  sotilaat ~1.8 — vihollisen armeijaa vastaan harjoittelu ei opettanut armeijan rakentamista (pitkän horisontin
  credit-ongelma; Outpostin +3 cap on NÄKYVISSÄ slotissa 4 mutta verkko ei opi sen arvoa).

## 3. Diagnoosi: kaksi rakenteellista ongelmaa

**Ongelma 1 — armeija jää rakentamatta (välitön pullonkaula, korjattavissa ilman täysremonttia).**
Attack-primitiivi toimii; AI ei vain rakenna sotilaskatto-infraa (Outpost), koska scaffold + litteä
negatiivinen Outpost-kandidaatti + puuttuva cap-gain-signaali ohjaavat cap-1:een. Self-playssa molemmat
ovat cap-1-kyhjääjiä → **leiriytyminen on tasapaino.**

**Ongelma 2 — katkaisu-voittoehto on näkymätön (syvempi katto, vaatii edustuskorjauksen).**
Spatiaalisesti sokea policy + maastopohjainen kohdevalinta ei voi nähdä artikulaatiopisteitä. Tämä on
se mihin koko *peli* perustuu.

---

## 4. Suositus: mitä kokeillaan (halvimmasta syvimpään)

### Koe G (suositus #1, halpa, jo rakennettu): `--vs-hard-frac`
Self-play opettaa vain itsensä voittamista → cap-1-kyhjäys. **Hard-botti fieldaa 7 sotilaan
iskuvoiman** → se yliajaa kyhjääjän → **pakottaa policyn oppimaan puolustuksen + Outpostit +
vastahyökkäyksen.** Lisäksi hard osaa katkaista → AI näkee esimerkkejä. Pidetään erillinen held-out-eval.
- Konfig: warm-start exp-A, **EI aggressio-Φ:tä** (se haittasi), spatial value, sims 96,
  `--vs-hard-frac 0.5`, 120 iter, cap 120. Verrataan ~33 %:iin.

### Koe H (jos G lupaava mutta cap-signaali puuttuu): paljasta armeija-potentiaali kandidaateissa
Lisää Outpostin `soldier_cap_gain` ja nykyinen sotilaskatto-headroom localVeciin, jotta net oppii
Outpost→armeija-arvon. **Koskee jaettua candidates-koodia → rikkoo parityn → golden-re-export + parity.**

### Koe I (syvä, tutkimuksen #1 vipu): katkaisu-tietoinen spatiaalinen policy
Anna policylle per-ruutu spatiaaliset piirteet ja **katkaisuarvo** (montako vihollisruutua irtoaa jos
otan tämän) → se näkee voittoehdon. Monipäiväinen, rikkoo julkaistun weights.ts:n. Todennäköisesti
ainoa reitti yli ~50 %:n kohti 70 %:a.

### Sivuhuomio: koulutus 14×12:lla (tai kokosekoituksella)
Koska AI on huonompi isolla kartalla ja se on pelin oletus, **siirrä treeni/benchmark 14×12:een** (tai
sekoitukseen 12×12…22×14). Halpa, deploy-relevantti.

---

## 5. Vastaus alkuperäisiin kysymyksiin
- **"Miten AI voi voittaa 30 % vaikka ottaa vähän ruutuja?"** Keskiarvo on harhaa: voitoissa se pitää
  ~45 %, mutta voittaa vain ~1.5 sotilaalla **kun vihollinen on jo romahtanut** (katkennut/konkurssi).
  Eli **kyllä, vastustaja häviää usein omaan heikkouteensa** — me emme murra puolustettua HQ:ta.
- **"Suosiiko se taloutta?"** Se rakentaa paljon taloutta JA yrittää hyökätä, mutta **ei rakenna
  armeija-infraa** (Outpost ~0 %) → jää 1 sotilaaseen. Syy on rakenteellinen (scaffold + cap-1-ansa),
  ei pelkkä reward-paino.
- **"Kuinka monimutkainen peli on?"** Mekaniikat ovat yksittäin yksinkertaisia, mutta **kaksi
  spatiaalista ydintä (HQ-katkaisu + voiman keskitys) tekevät siitä graafipelin** jota aggregoitu
  vektori + heuristinen kohdevalinta ei riitä pelaamaan. Tämä vahvistaa edustuskaton — ei arvauksena
  vaan mekaniikasta johdettuna.

## 6. Exp I -tulokset (2026-06-03, yön yli) — edustus korjattu, mutta koulutusdynamiikka kaatuu

Toteutettiin koko spatiaalinen/katkaisu-tietoinen policy (AZ-only, parity 8/8 säilyi, live-peli koskematon):
`spatial::offensive_cut_value` (yksikkötestattu), `policy_spatial` (6 piirrettä: cut-arvo, HQ-läheisyys,
is-enemy-HQ, oma-cut-haavoittuvuus, vihollisnaapuri-osuus, owner-is-enemy), `--spatial-policy` MCTS-prioreihin
+ self-playhin, `warmstart_spatial`-siirto (exp-A → [69,24,16,1], 6 spatiaalipainoa 0 → init == exp-A), ja
best-checkpoint-tallennus.

**Kaksi ajoa, molemmat EPÄONNISTUIVAT voittamaan exp-A:n:**
- **az10 (cold-start, pure self-play):** juuttui ~5% / 115 iter — satunnainen policy ei opi perusasioita.
- **az11 (warm-start, pure self-play, 14×12):** lähti 31–40%, **valui alas ~12%:iin** 410 iter aikana, ei palautunut.
  Best = kouluttamaton warm-start ≈ exp-A (champ_probe **31% legit**).
- **az12 (warm-start + --vs-hard-frac 0.75 + lr 5e-4, ankkurointi):** piti ~22–25% (gen0–20), **valui ~10%:iin**.
  Best gen15 champ_probe **21.5% legit** — exp-A:n ALLE.

**Johtopäätös (luja):** Spatiaalinen **edustus on rakennettu, oikein ja toimii** (cut-feature todennettu,
champ_probe ajaa sen). MUTTA **self-play-koulutusdynamiikka valuttaa policyn passiivisuuteen** (draw-attractor:
jaettu, ajautuva policy degeneroituu yhdessä → timeoutit 60–65%, tileFrac → 0.09). **Edustus on välttämätön
mutta ei riittävä** — ja **kumpikaan, puhdas self-play TAI 75% kiinteä-vastustaja-ankkurointi, ei estä valumista.**
Paras deployattava malli on yhä **exp-A (31% @ 14×12, 33% @ 12×12).**

**Seuraava vipu = koulutuksen VAKAUS (ei lisää laskentaa samalla loopilla):**
1. **KL/trust-region-ankkurointi warm-startiin** — rankaise policya kun se ajautuu kauas exp-A:sta → ei kävele
   passiivisuuteen. (PPO-tyylinen klippi tai KL-sakko policy_train.rs:ään.)
2. **Value-verkon ankkurointi/jäädytys** — value co-trainaa ja saattaa romahtaa "kaikki on tasapeli" -tilaan,
   ruokkien MCTS:n passiivisuutta. Kokeile jäädytettyä tai hitaammin päivittyvää valuea.
3. **BC-lämmittely + erittäin lempeä RL** — pidä policy lähellä exp-A:ta, nyörää vain kevyesti.
4. Vasta jos vakaus ratkeaa: katso oppiiko se käyttämään cut-näköä (sen piti olla koko pointti).

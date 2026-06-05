# Palkkiosuunnittelu — AlphaZero AI

_Laadittu 2026-06-02. Käyttäjän antamat positiiviset/negatiiviset signaalit + niiden
kytkentä (a) AlphaZeron **potentiaalipohjaiseen shaping-termiin** Φ(s) ja (b)
**spatiaalisiin piirteisiin** ja value-verkon arviointiin._

## Periaate (miksi näin)

AlphaZero oppii **varsinaisen strategian pelkästä lopputuloksesta** (voitto +1 /
tappio −1 / tasapeli 0). MCTS + value-verkko hoitavat pitkän horisontin
credit-assignmentin — juuri sen, mihin GA ei pysty. Käyttäjän signaaleja EI siis
leivota kovaksi rewardiksi (se vinouttaisi optimia, Ng 1999). Sen sijaan:

- **Potentiaalipohjainen shaping** `F(s,s') = γΦ(s') − Φ(s)` säilyttää optimaalisen
  politiikan todistetusti (Ng, Harada, Russell 1999). Φ(s) on "kuinka hyvältä tämä
  asema näyttää". Käyttäjän signaalit määrittävät Φ:n — ne vain **nopeuttavat
  oppimista**, eivät muuta sitä mikä on optimaalista. Monimutkaisetkin signaalit ovat
  tässä turvallisia.
- **Spatiaaliset piirteet** (vaihe `representation`): osa signaaleista on
  paikkasidonnaisia (etäisyys vihollisen tukikohtaan, oman alueen katkaisu-alttius,
  rintamapaine) → ne menevät verkon syötteeseen, eivät rewardiin.
- **Tapahtumat hoituvat haun kautta**: "tukikohdan valtaus → voitto" näkyy
  value-verkossa automaattisesti, koska forward-model toteuttaa säännön.

## Käyttäjän positiiviset signaalit → kytkentä

| # | Signaali (käyttäjä) | Suure | Lähde |
|---|---|---|---|
| P1 | Talouskasvua tapahtuu | net income, tuottava pinta-ala, solvenssi | `mean_net_income_norm`, `mean_productive_area`, `mean_solvency` ✓ |
| P2 | Laajentuminen tapahtuu | dominaation eteneminen (tile_frac/0.70) | `mean_domination_progress` ✓ |
| P3 | Ruutuja enemmän kuin vihollisella | ruutujohto (signed) | `mean_tile_lead` ✓ |
| P4 | Resurssit riittävät kasvusuunnitelmaan (rakentaa lisää tuottavaa) | solvenssi + tuottava pinta-ala | `mean_solvency`, `mean_productive_area` ✓ |
| P5 | Alueen puolustus onnistuu (sotilaat/tukikohta) | ei menetä ruutuja hyökkäyksessä; sotilasjohto | `tile_loss`≈0, `mean_military_lead` ✓ |
| P6 | Vihollisen alueen valtaus | vallatut viholliset ruudut | `enemy_tiles_conquered` ✓ |
| P7 | Hyödyllisen alueen valtaus (tehdas/farmi) | vallatut viholliset **rakennukset** | `enemy_buildings_captured` ✓ |
| P8 | Vihollisen alueen **katkaisu** (paljon menetettyjä sotilaita/rakennuksia) | katkaisulla saadut ruudut | `tiles_gained_via_cut` ✓ |
| P9 | Vihollisen talous menee huonoon kuntoon | viholliseen nähden talous/varallisuusjohto | `mean_income_lead`, `mean_wealth_lead` ✓ |
| P10 | Hyökätä mahd. lähelle vihollisen tukikohtaa (vallata sieltä ruutuja) | **etäisyys vihollisen HQ:hon** vallatuissa ruuduissa | **UUSI** → spatiaalinen piirre + tactical-bonus |
| P11 | Tukikohdan valtaus (= voitto) | vallatut viholliset HQ:t; itse voitto | `enemy_hqs_captured`, terminal `won` ✓ |

## Käyttäjän negatiiviset signaalit → kytkentä

| # | Signaali (käyttäjä) | Suure | Lähde |
|---|---|---|---|
| N1 | Alueiden menetys | ruutujen putoaminen alkutason alle | `tile_loss` (w_tile_loss) ✓ |
| N2 | Omien sotilaiden/joukkojen kuolema | **omat menetetyt sotilaat** | **UUSI** telemetria-kenttä `own_soldiers_lost` |
| N3 | **Oman alueen katkaisu** | oma menetetty alue katkaisun takia | **UUSI** telemetria-kenttä `own_tiles_lost_via_cut` |
| N4 | Hidas talouskasvu / lasku | matala/negatiivinen income-trendi | `mean_net_income_norm` matala; lisää **income-trendi Δ** |
| N5 | Potentiaalin käyttämättä jättäminen | käyttämätön työvoima / idlet resurssit / rakentamatta jättäminen | **UUSI** piirre `idle_potential` (työvoima-vajaus + ylijäämäresurssi ilman rakentamista) |
| N6 | Vihollinen edellä taloudessa tai alueessa | negatiiviset johdot (jo signed) | `mean_tile_lead<0`, `mean_income_lead<0`, `mean_wealth_lead<0` ✓ |

## Potentiaali Φ(s) AlphaZeron shapingiin (luonnos)

Φ(s) on rajattu välille [−1, 1] ja lasketaan nykyisestä tilasta (ei trajektorin
keskiarvosta). Alustavat painot tunnistetaan empiirisesti (train→benchmark→säädä):

```
Φ(s) =  w1 * domination_progress           # P2  laajentuminen
      + w2 * tile_lead                       # P3,N6  ruutujohto (signed)
      + w3 * econ_health                     # P1,P4  income+solvenssi+tuottava ala
      + w4 * (wealth_lead + income_lead)/2   # P9,N6  talousjohto (signed)
      + w5 * military_lead                    # P5  puolustuskyky (signed)
      - w6 * idle_potential                   # N5  käyttämätön potentiaali
      - w7 * proximity_to_own_hq_threat       # oman HQ:n uhka (kytkee N3:een)
```

Shaping per siirto: `r_shape = γ Φ(s') − Φ(s)`. Tapahtumat (valtaus, katkaisu,
HQ-valtaus, tappaminen, oman alueen menetys) **eivät** mene Φ:hen vaan ne joko
(a) näkyvät value-verkossa haun kautta, tai (b) jäävät kevyiksi
tactical-bonuksiksi self-play-datan rikastamiseen — pidetään pieninä, ettei
optimi vinou.

## Spatiaaliset piirteet (vaihe `representation`) joita nämä signaalit edellyttävät

- **Etäisyys vihollisen tukikohtaan** per ehdokasruutu (P10) — ohjaa hyökkäämään kohti HQ:ta.
- **Etäisyys omaan tukikohtaan + katkaisu-alttius** (N3) — näkee oman HQ-yhteyden kapeikot.
- **Rintamapaine / naapuruston omistus** (P5, N1) — puolustustarve.
- **Ruudun hyödyllisyys** (rakennustyyppi: tehdas/farmi) (P7) — kohdevalinta.

## Tehtävät tästä

1. Lisää telemetriaan `own_soldiers_lost`, `own_tiles_lost_via_cut`, income-trendi Δ
   (cp-ai `SeatTelemetry`) — parity-vaikutus tarkistettava.
2. Lisää spatiaaliset piirteet (etäisyys HQ:hon jne.) vaiheessa `representation`.
3. Toteuta Φ(s) ja `γΦ(s')−Φ(s)`-shaping AlphaZero-self-playhin (vaihe `trainloop`).
4. Viritä painot empiirisesti: vertaa win-rate vs hard, ei raakaa fitnessiä.

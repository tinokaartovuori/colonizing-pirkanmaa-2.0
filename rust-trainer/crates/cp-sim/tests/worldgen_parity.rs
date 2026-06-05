//! Worldgen parity: a fixed seed must reproduce the exact terrain the TS engine
//! produced, tile-by-tile in column-major (`index = x*height + y`) order.
//!
//! The expected terrain strings were extracted from the golden traces in
//! `rust-trainer/golden/trace-<seed>.json` (the `map[].type` field, which is in
//! `getTiles()` = generation order). Each char: G=Grassland, F=Forest,
//! A=Abundant Forest, M=Mountain, R=River. The golden `map` is the FINAL game
//! state, so buildings differ — but terrain type never changes after gen, so
//! the type string is a faithful gen fingerprint.

use cp_sim::{Game, TileType};

fn type_char(t: TileType) -> char {
    match t {
        TileType::Grassland => 'G',
        TileType::Forest => 'F',
        TileType::AbundantForest => 'A',
        TileType::Mountain => 'M',
        TileType::River => 'R',
    }
}

/// `(seed, width, height, expected_terrain_column_major)` for the whole golden suite.
const GOLDEN: &[(u32, i32, i32, &str)] = &[
    (1, 12, 12, "GGGRGGGGGMAGGGRRGGGGMGGGGGRGGGGGGGGGFRRFFGFAGGGFMRGFFFFAGGMFGRFFFFMFMGFFGRRFFFFFGGFFGGRGGGGFFFFFGRRGGGGFFGGGFRFGMGGFFFGARRFGGGGFFFGGRFFFGFGGFFGG"),
    (123, 18, 14, "AGGFFFGGFFAFFMGGFFFGGGGFGAGMGGGFFGGMGFGGGGFFGGGGGFFGFAGGMFFGGGFFGFFFFFFGFGGGFGMFGFFFMRRRRGGRRRGFFFRRGMRRRRFRRRRMFGFFFGFMMMGGRRGFFGFGFFFAGGGGFFFFGFGFGFGGGGFFFFFGGFFFGGGGFFGFFGGFGMGGMGGFFAGFGGGGGGGFGGGGFGFGGGGGFFGGFFFFFFGGGAFGGGFFFGGFFGGGGGGGFFFGFFGMFGGG"),
    (13, 14, 12, "GMGGGGFFFGGGGGGGGMFFFGGGGGGAFMFFFFFFGFGFFFFGGGMFGGMFFFFGGFFGGGFFFGGGGGMGGGMFFGGGGFFFGGGFFFFFFFFFGGGGMFFFGFFFGGGGGFFFFFFFGGGGGFAFGGGGRRGGRRRRRRFGFRRRRGGFARRRGGGGGGGAGGFF"),
    (2, 12, 12, "FGFMFGFFFFFFMFFFGFFGGFFGFFFFAFFGGFFRRRRGRRRFGRRRRFRRRGRRRRGGGGFFGFFFGGGGAGFFAAFFMFGGGFGFFGGFFFFMFGFFFGGFFFGGGGFFFFGFFGFGGGFFFFGFFFGGMGMGGFFFGFFM"),
    (256, 20, 15, "GGGGGGGGFFFMGGGGGFFFGMAGFFGGGGGGFFFFFFGGFRRRRFFFFFFRRRRRRGGGGGFGRRRFGGGGGGGFRRRRFFFGFFFFMGRRFFFFFFAGFFFFGFFFFFFGAGFFFFFMFAFGGFFFFFFFFMFGGGFGFFFFFFFFAFGAGGFFFFFGFFFFGFGGGGFFFFGGFGGGFFFFFFFFGGFFFGGGFFFFFFFGGFFFGGFFFGGGFMFFFFFFFGGFGFFFMFFAFFGFGFFGFFGGFFFFFFFMGGGGGGGFFFFAMFFGGGGGGGGGGGFFFGGGGGGGGGGGGGGG"),
    (42, 16, 14, "FFFMFGMFRRFFFGFFFFGGGMRFFFGFFFFMGFMGRFFFFGGFFFFGFFRGFFGGFMFFGGFFRRFFGGMGFGGGFFFRFFFGFFFFGGFFGRFGMAFAFFGGGGGRRGGGFMFGFFGFGGRGGGGGGGGMFMMFRGFGFGGGGGFFFFRGFFGGGFGFGGFMRRFFAGAMFFAFFFGRFFGGGFMAGFFFGRRMGMGGGFFFFGGGRMGGGGGGGFGFGMRR"),
    (7, 14, 12, "GGFFGGGRRGGGGGAFFGFFRFGGGGGFFGGRRGFMMFFGGGFRFFGMFFFFGGGRRGFFFFFGGGGGRFFFGFFFGFFFRFFFGGAGGGFRRFFAGGGGGGFRGGFFGGGGGGRRFGAFGAGGGGRFFGFFFGGGGRRFGFFFFFFFGRGGFMGGFFFGFRGFFFGG"),
    (99, 16, 14, "GGRRRRRFRRRGAGRRRGFFRRRFFMGGRGFFFGGFFFGGFGMFFMGGGFGGGGFFFGFFFFFFFFAGFFGMFFGFMFGFGGGGFFFFFAMGFFFGGGGGFFFGFFFFFGGGGFGFFFFFFMFGGGGFFFGGFFFFFAGGFFFGGGFFGFFGGGFFFGGFFFGFFFGGFAFAGFFFGGFMGGGGMFGFFGGFGFGGFGFFGMFFGGGFGMMGFFFFGGGGFFMG"),
];

#[test]
fn worldgen_matches_golden_terrain_for_all_seeds() {
    for &(seed, w, h, expected) in GOLDEN {
        let mut game = Game::new(w, h, &["P1", "P2"]);
        game.generate_map(w, h, seed);

        let got: String = game.get_tiles().iter().map(|t| type_char(t.tile_type)).collect();
        assert_eq!(
            got.len(),
            expected.len(),
            "seed {seed}: tile count mismatch ({}x{})",
            w,
            h
        );
        assert_eq!(got, expected, "seed {seed}: terrain mismatch\n got: {got}\nwant: {expected}");
    }
}

#[test]
fn worldgen_column_major_order() {
    // Confirm the tile at storage index i has coordinate (i/height, i%height).
    let (w, h) = (12, 12);
    let mut game = Game::new(w, h, &["P1", "P2"]);
    game.generate_map(w, h, 1);
    for (i, t) in game.get_tiles().iter().enumerate() {
        assert_eq!(t.x, (i as i32) / h);
        assert_eq!(t.y, (i as i32) % h);
    }
}

#[test]
fn worldgen_is_deterministic() {
    let mut a = Game::new(12, 12, &["P1", "P2"]);
    a.generate_map(12, 12, 12345);
    let mut b = Game::new(12, 12, &["P1", "P2"]);
    b.generate_map(12, 12, 12345);
    let ta: Vec<TileType> = a.get_tiles().iter().map(|t| t.tile_type).collect();
    let tb: Vec<TileType> = b.get_tiles().iter().map(|t| t.tile_type).collect();
    assert_eq!(ta, tb);
}

#[test]
fn worldgen_places_exactly_one_mikontalo() {
    use cp_sim::BuildingType;
    let mut game = Game::new(12, 12, &["P1", "P2"]);
    game.generate_map(12, 12, 1);
    let mik: Vec<usize> = game
        .get_tiles()
        .iter()
        .enumerate()
        .filter(|(_, t)| matches!(&t.building, Some(b) if b.kind == BuildingType::Mikontalo))
        .map(|(i, _)| i)
        .collect();
    // Golden trace-1 has its Mikontalo at tile index 13.
    assert_eq!(mik, vec![13]);
}

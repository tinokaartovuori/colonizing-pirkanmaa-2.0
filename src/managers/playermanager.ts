// Port of DAL/playermanager.{h,cpp}.

import { PlayerBase, PlayerConfig } from '../model/player';
import { IObjectManager } from '../model/base';

export class PlayerManager {
  private playerIndex_ = 0;
  private players_: PlayerBase[] = [];
  private lostPlayers_: PlayerBase[] = [];
  private roundsPlayed_ = -1;

  /**
   * Accepts either a list of plain names (all human — used by the tests) or a
   * list of {@link PlayerConfig} objects that may flag a player as a CPU.
   */
  constructor(players: Array<string | PlayerConfig>, objectManager: IObjectManager) {
    for (let i = 0; i < players.length; i++) {
      const cfg = players[i];
      if (typeof cfg === 'string') {
        this.players_.push(new PlayerBase(cfg, i + 1, objectManager));
      } else {
        this.players_.push(new PlayerBase(cfg.name, i + 1, objectManager, cfg.difficulty ?? 'human'));
      }
    }
  }

  /** True once every remaining player is computer-controlled (no human left). */
  hasNoHumanLeft(): boolean {
    return this.players_.length > 0 && this.players_.every((p) => p.isCpu());
  }

  getCurrentPlayer(): PlayerBase {
    return this.players_[this.playerIndex_];
  }

  getPlayers(): PlayerBase[] {
    return this.players_;
  }

  getLostPlayers(): PlayerBase[] {
    return this.lostPlayers_;
  }

  changeTurn(): void {
    this.playerIndex_++;
    if (this.playerIndex_ >= this.players_.length) {
      this.playerIndex_ = 0;
    }
    if (this.playerIndex_ === 0) {
      this.roundsPlayed_++;
    }
  }

  setPlayerAsLost(lostPlayer: PlayerBase, currentPlayer: PlayerBase | null = null): void {
    this.lostPlayers_.push(lostPlayer);

    if (currentPlayer !== null && lostPlayer.getPlayerNum() < currentPlayer.getPlayerNum()) {
      this.playerIndex_--;
    }

    const idx = this.players_.indexOf(lostPlayer);
    if (idx !== -1) this.players_.splice(idx, 1);
  }

  getRoundsPlayed(): number {
    return this.roundsPlayed_;
  }

  /**
   * Restore turn bookkeeping from a saved game: remove the players that had already
   * lost, set the rounds counter, and point the turn at the saved current player.
   * Players were reconstructed in their original order, so player numbers match.
   */
  restore(currentPlayerNum: number, roundsPlayed: number, lostPlayerNums: number[]): void {
    for (const num of lostPlayerNums) {
      const lost = this.players_.find((p) => p.getPlayerNum() === num);
      if (lost) {
        this.lostPlayers_.push(lost);
        this.players_.splice(this.players_.indexOf(lost), 1);
      }
    }
    this.roundsPlayed_ = roundsPlayed;
    const idx = this.players_.findIndex((p) => p.getPlayerNum() === currentPlayerNum);
    this.playerIndex_ = idx >= 0 ? idx : 0;
  }
}

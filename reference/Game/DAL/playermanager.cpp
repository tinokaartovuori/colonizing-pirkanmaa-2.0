/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: playermanager.cpp, see playermanager.h for more info         #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "playermanager.h"


namespace Student {
PlayerManager::PlayerManager(std::vector<std::string> players,
               std::shared_ptr<Course::iObjectManager> objectmanager):
               playerIndex_(0),
               objectManager_(objectmanager),
               lostPlayers_({}),
               roundsPlayed_(-1)
{
    for (int i=0; i < (int)players.size(); ++i) {
            std::shared_ptr<Course::PlayerBase> playerPtr =
                    std::make_shared<Course::PlayerBase>
                    (players.at(i), i+1, objectManager_);
            players_.push_back(playerPtr);
    }
}


std::shared_ptr<Course::PlayerBase> PlayerManager::getCurrentPlayer()
{
    return (players_.at(playerIndex_));
}


std::vector<std::shared_ptr<Course::PlayerBase>> PlayerManager::getPlayers()
{
    return players_;
}


std::vector<std::shared_ptr<Course::PlayerBase> > PlayerManager::getLostPlayers()
{
    return lostPlayers_;
}


void PlayerManager::changeTurn()
{
    ++playerIndex_;

    if (playerIndex_ >= (int)players_.size()) {
        playerIndex_ = 0;
    }
    if (playerIndex_ == 0) {
        ++roundsPlayed_;
    }
}


void PlayerManager::setPlayerAsLost(std::shared_ptr<Course::PlayerBase> lostPlayer,
                                std::shared_ptr<Course::PlayerBase> currentPlayer)
{
    lostPlayers_.push_back(lostPlayer);

    //Prevents the player index increasing by two if a previous round player lost
    if (currentPlayer != nullptr &&
            lostPlayer->getPlayerNum() < currentPlayer->getPlayerNum())
    {
        --playerIndex_;
    }

    //Player is removed from the players that can still play
    for(std::vector<std::shared_ptr<Course::PlayerBase>>::iterator
                              it = players_.begin(); it != players_.end();)
    {
        if (*it == lostPlayer){
            it = players_.erase(it);
            break;
        } else {
            ++it;
        }
    }

}


int PlayerManager::getRoundsPlayed()
{
    return roundsPlayed_;
}


}






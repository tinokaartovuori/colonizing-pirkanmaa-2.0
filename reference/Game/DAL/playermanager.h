/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: playermanager.h, header to PlayerManager-class               #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef PLAYERMANAGER_H
#define PLAYERMANAGER_H

#include "Core/playerbase.h"
#include "Core/resourcemaps.h"

#include <vector>
#include <string>
#include <memory>


namespace Student {

/**
 * @brief PlayerManager manages the players in the game. It tracks which player
 * is in turn and which have lost.
 */

class PlayerManager
{
public:

    /**
     * @brief Constructor for the class. The constructor creates the shared
     *        pointers (PlayerBase) pointing to the players
     * @param players is a vector containing the player names in string type
     * @param objectmanager points to the ObjectManager.
     * @post Exception Guarantee: No guarantee.
     */
    PlayerManager(std::vector<std::string> players,
                     std::shared_ptr<Course::iObjectManager> objectmanager);


    /**
     * @brief Returns a shared pointer to the player whose turn it is
     * @return Shared pointer to the player in turn
     * @post Exception Guarantee: No guarantee.
     * @exception Out of range is possible in theory, though the index cannot be
     *            set to be too high
     */
    std::shared_ptr<Course::PlayerBase> getCurrentPlayer();


    /**
     * @brief Returns a vector of shared pointers to the players
     *        who are still in the game and haven't lost
     * @return Vector of the shared pointer to the players still in game
     * @post Exception Guarantee: No-throw.
     */
    std::vector<std::shared_ptr<Course::PlayerBase>> getPlayers();


    /**
     * @brief Returns a vector of shared pointers to the players
     *        who have lost
     * @return Vector of the shared pointers to the players who have lost
     * @post Exception Guarantee: No-throw.
     */
    std::vector<std::shared_ptr<Course::PlayerBase>> getLostPlayers();


    /**
     * @brief Changes the index which keeps track which player is in turn.
     *        The function also keeps track of how many turns there's been in
     *        the game.
     * @post Exception Guarantee: No-throw.
     */
    void changeTurn();


    /**
     * @brief Sets player as lost. This is done by removing the player from
     *        the players_ vector and adding the player into lostPlayers_ vector
     * @param Shared pointer to the player to be set as lost
     * @post Exception Guarantee: Strong
     */
    void setPlayerAsLost(std::shared_ptr<Course::PlayerBase> lostPlayer,
                     std::shared_ptr<Course::PlayerBase> currentPlayer = nullptr);


    /**
     * @brief Returns an integer of the rounds there's been in the game
     * @return Integer of the rounds played
     * @post Exception Guarantee: No-throw.
     */
    int getRoundsPlayed();


private:
    int playerIndex_; //Index to the current player in turn (players_ vector)

    std::weak_ptr<Course::iObjectManager> objectManager_;

    std::vector<std::shared_ptr<Course::PlayerBase>> lostPlayers_; //Players that have lost
    std::vector<std::shared_ptr<Course::PlayerBase>> players_; //Players that are playing

    int roundsPlayed_; //How  many rounds there's been in the game

};

}


#endif // PLAYERMANAGER_H

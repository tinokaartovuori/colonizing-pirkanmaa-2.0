/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: gamesettingsmanager.h, header to the GameSettingsManager-class #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef GAMESETTINGSMANAGER_H
#define GAMESETTINGSMANAGER_H

namespace Student {

/**
 * @brief The GameSettingsManager class is an interface for different methods
 *        to access various game settings variaables
 */
class GameSettingsManager
{
public:


    GameSettingsManager();

    /**
     * @brief GameSettingsManager constructor
     * @param mapGridSize is an integer of
     *        how many pixels one map grid (tile) is wide and tall
     * @param menuGridSize is an integer of
     *        how many pixels one menu grid is wide and tall
     * @param mapWidth is an integer of how many pixels the map has horizontally
     * @param mapHeight is an integer of how many pixels the map has vertically
     * @param menuWidth is an integer of how many pixels wide the menu is
     * @param menuHeight is an integer of how many pixels tall the menu is
     *
     * @post Exception guarantee: Strong
     */

    GameSettingsManager(int mapGridSize,
                        int menuGridSize,
                        int mapWidth,
                        int mapHeight,
                        int menuWidth,
                        int menuHeight);


    /**
     * @brief getMapGridSize
     * @return an integer of
     *         how many pixels one map grid (tile) is wide and tall
     * @post Exception guarantee: no-throw
     */
    int getMapGridSize();


    /**
     * @brief getMenuGridSize
     * @return an integer of
     *         how many pixels one menu grid is wide and tall
     * @post Exception guarantee: no-throw
     */
    int getMenuGridSize();


    /**
     * @brief getMapWidth
     * @return an integer of how many tiles the map has horizontally
     * @post Exception guarantee: no-throw
     */
    int getMapWidth();


    /**
     * @brief getMapHeight
     * @return an integer of how many pixels the map has vertically
     * @post Exception guarantee: no-throw
     */
    int getMapHeight();


    /**
     * @brief getMapHeight
     * @return an integer of how many pixels wide the menu is
     * @post Exception guarantee: no-throw
     */
    int getMenuWidth();


    /**
     * @brief getMapHeight
     * @return an integer of how many pixels tall the menu is
     * @post Exception guarantee: no-throw
     */
    int getMenuHeight();


    /**
     * @brief getMapHeight
     * @return an integer of how many tiles the map has horizontally
     * @post Exception guarantee: no-throw
     */
    int getMapGridWidth();


    /**
     * @brief getMapHeight
     * @return an integer of how many tiles the map has vertically
     * @post Exception guarantee: no-throw
     */
    int getMapGridHeight();


private:
    int mapGridSize_;  //The width and height of one map tile in pixels
    int menuGridSize_; //The width and height of one menu tile in pixels
    int mapWidth_; //How many pixels the map has horizontally
    int mapHeight_; //How many pixels the map has vertically
    int menuWidth_; //How many pixels wide the menu is
    int menuHeight_; //How many pixels tall the menu is

};
}
#endif // GAMESETTINGSMANAGER_H

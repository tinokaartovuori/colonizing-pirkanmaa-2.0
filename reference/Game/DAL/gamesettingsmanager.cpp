/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: gamesettingsmanager.cpp, see gamesettingsmanager.h for more info  #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "gamesettingsmanager.h"

namespace Student {

GameSettingsManager::GameSettingsManager()
{
}

GameSettingsManager::GameSettingsManager(int mapGridSize,
        int menuGridSize,
        int mapWidth,
        int mapHeight,
        int menuWidth,
        int menuHeight):
    mapGridSize_(mapGridSize),
    menuGridSize_(menuGridSize),
    mapWidth_(mapWidth),
    mapHeight_(mapHeight),
    menuWidth_(menuWidth),
    menuHeight_(menuHeight)
{
}

int GameSettingsManager::getMapGridSize()
{
    return mapGridSize_;
}

int GameSettingsManager::getMenuGridSize()
{
    return menuGridSize_;
}

int GameSettingsManager::getMapWidth()
{
    return mapWidth_;
}

int GameSettingsManager::getMapHeight()
{
    return mapHeight_;
}

int GameSettingsManager::getMapGridWidth()
{
    return mapWidth_/mapGridSize_;
}

int GameSettingsManager::getMapGridHeight()
{
    return mapHeight_/mapGridSize_;
}

int GameSettingsManager::getMenuWidth()
{
    return menuWidth_;
}

int GameSettingsManager::getMenuHeight()
{
    return menuHeight_;
}

}

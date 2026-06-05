/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: objectmanager.cpp, see objectmanager.h for more info         #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "objectmanager.h"
#include "iostream"
#include <QDebug>

namespace Student {

ObjectManager::ObjectManager()
{
}


void ObjectManager::setGameScene(std::shared_ptr<GameScene> gs)
{
    gameScene_ = gs;
}


std::shared_ptr<GameScene> ObjectManager::getGameScene()
{
    return gameScene_.lock();
}


void ObjectManager::addTiles(
        const std::vector<std::shared_ptr<Course::TileBase>> &tiles)
{
    for (const auto& tile : tiles) {
        tiles_.push_back(tile);
    }
}

void ObjectManager::replaceTile(std::shared_ptr<Course::TileBase> oldTile,
                               std::shared_ptr<Course::TileBase> newTile)
{
    bool removed = false;
    for(std::vector<std::shared_ptr<Course::TileBase>>::iterator
                              it = tiles_.begin(); it != tiles_.end();)
    {
        if (*it == oldTile){
            it = tiles_.erase(it);
            removed = true;
            break;
        } else {
            ++it;
        }
    }

    if (removed) {
        tiles_.push_back(newTile);
        newTile->setOwner(oldTile->getOwner());
        for (auto unit : oldTile->getUnits()) {
            unit->addParentTile(newTile);
            newTile->addUnit(unit);
        }
    }
    else {
        qDebug()<<"Error, tile to be replaced was not found.";
    }
}


std::shared_ptr<Course::TileBase> ObjectManager::getTile
                                (const Course::Coordinate &coordinate)
{
    for (const auto& tile : tiles_) {
        if (tile->getCoordinate() == coordinate) {
            return tile;
        }
    }

    return nullptr; //If the tile wasn't found a nullptr will be returned
}


std::vector<std::shared_ptr<Course::TileBase> > ObjectManager::getTiles()
{
    return tiles_;
}


void ObjectManager::setHoverBorder(
        const std::shared_ptr<Student::MouseHoverBorder> border)
{
    hoverBorder_ = border;
}


void ObjectManager::setClickedTileBorder(std::shared_ptr<Course::TileBase> tile)
{
    clickedTileBorder_ = std::make_shared<Student::ClickedTileBorder>
            (tile->getCoordinate(), 1, 1,
             gameEventHandler_.lock(), shared_from_this());

   clickedTileBorder_->setImageFiles(ImageVectors::CLICKEDTILEBORDER);
   gameScene_.lock()->drawItem(clickedTileBorder_);

}


std::shared_ptr<Student::ClickedTileBorder> ObjectManager::getClickedTileBorder()
{
    return clickedTileBorder_;
}


void ObjectManager::removeClickedTileBorder()
{
    if (clickedTileBorder_ != nullptr) {
        gameScene_.lock()->removeItem(clickedTileBorder_);
    }

    clickedTileBorder_ = nullptr;
}


std::shared_ptr<Student::MouseHoverBorder> ObjectManager::getBorderTile()
{
    return hoverBorder_;
}


void ObjectManager::addDALS(
        const std::shared_ptr<Course::iGameEventHandler> gameeventhandler,
        const std::shared_ptr<Student::iMenuObjectManager> menuobjectmanager,
        const std::shared_ptr<Student::GameSettingsManager> gamesettingsmanager)
{
    gameEventHandler_ = gameeventhandler;
    menuObjectManager_ = menuobjectmanager;
    gameSettingsManager_ = gamesettingsmanager;
}


std::vector<std::shared_ptr<Course::TileBase>> ObjectManager::getHqConnectedTiles
                                     (std::shared_ptr<Course::PlayerBase> player)
{
    std::vector<std::shared_ptr<Course::TileBase>> tiles;

    //Headquarters exist
    if (getHqTile(player) != nullptr) {
        tiles.push_back(getHqTile(player));

        //When all tiles have been found the for-loop ends
        for (int i = 0; i < (int)tiles.size(); ++i) {
            /*Loops all neighbouring tiles and adds them into tiles-vector
             *if the player owns the tile*/
            for (auto neighbour: tiles.at(i)->getNeighbourFourTiles())
            {
                /*If the neighbour has already been added it will be skipped
                 *and won't be added*/
                if (std::find(tiles.begin(), tiles.end(),
                              neighbour) != tiles.end()) {
                    continue;
                }
                if (player == neighbour->getOwner()) {
                    tiles.push_back(neighbour);
                }
            }
        }
    }

    return tiles;
}


std::shared_ptr<Course::TileBase> ObjectManager::getHqTile
                            (std::shared_ptr<Course::PlayerBase> player)
{
    std::shared_ptr<Course::TileBase> HqTile;

    //Loops all player's objects
    for (auto object : player->getObjects()) {
        if (std::dynamic_pointer_cast<Course::TileBase>(object) != nullptr)
        {
            std::shared_ptr<Course::TileBase> tile =
                    std::dynamic_pointer_cast<Course::TileBase>(object);

            if (tile->getBuilding() != nullptr &&
                    tile->getBuilding()->getType() == "Headquarters")
            {
                //Tile with headquarters is found
                std::shared_ptr<Course::HeadQuarters> HQ =
                                std::dynamic_pointer_cast
                                <Course::HeadQuarters>(tile->getBuilding());
                //Only non-conquered headquarters tile is returned.
                //A conquered one doesn't count
                if (!HQ->isConquered()) {
                    HqTile = tile;
                    break;
                }
            }
        }
    }

    return HqTile;
}


std::vector<std::shared_ptr<Course::TileBase>> ObjectManager::getAvailableTiles()
{
    std::shared_ptr<Course::PlayerBase> player = gameEventHandler_.lock()
                                                    ->getCurrentPlayer();
    std::vector<std::shared_ptr<Course::TileBase>> availableTiles = {};

    for (auto obj : player->getObjects()) {
        if (std::dynamic_pointer_cast<Course::TileBase>(obj) != nullptr) {
            std::shared_ptr<Course::TileBase> tile =
                                   std::dynamic_pointer_cast<Course::TileBase>(obj);

            // Check if tile is already in the available tiles

            // Tile is owned by player and units can be placed
            if (tile->getOwner() == player and tile
                    ->hasOpponentHeadquarters(player))
            {
                if (std::find(availableTiles.begin(), availableTiles.end(), tile)
                                                        != availableTiles.end()) {
                    // tile is already in available tiles
                } else {
                    availableTiles.push_back(tile);
                }
            }

            if (tile->getType() == "River" and tile->getBuilding() == nullptr) {
                continue;
            } else {
                std::vector<std::shared_ptr<Course::TileBase>> neighbourTiles =
                        tile->getNeighbourFourTiles();
                for (std::shared_ptr<Course::TileBase> n_tile : neighbourTiles) {
                    if (std::find(availableTiles.begin(), availableTiles.end(),
                                  n_tile) != availableTiles.end()) {
                        // tile is already in available tiles
                        continue;
                    } else {
                        if (n_tile->hasOpponentHeadquarters(player)) {
                            availableTiles.push_back(n_tile);
                        }
                    }
                }
            }
        }
    }

    return availableTiles;
}


void ObjectManager::addBlockTileOverlays() {
    std::vector<std::shared_ptr<Course::TileBase>> availableTiles =
                                    getAvailableTiles();
    std::vector<std::shared_ptr<Course::TileBase>> blockedTiles = {};

    for (auto tile : getTiles()) {
        bool tileFound = false;
        for (auto a_tile : availableTiles) {
            if (tile == a_tile) {
                tileFound = true;
                break;
            }
        }
        if (!tileFound) {
            blockedTiles.push_back(tile);
        }
    }

    for (auto blockedTile : blockedTiles) {
        std::shared_ptr<Student::BlockedTile> overlay =
                std::make_shared<Student::BlockedTile>(
                                            blockedTile->getCoordinate(),
                                            1,
                                            1,
                                            gameEventHandler_,
                                            shared_from_this());
        overlay->setImageFiles(ImageVectors::BLOCKED_TILE);
        blockedTileOverlays_.push_back(overlay);
        gameScene_.lock()->drawItem(overlay);
    }
}


void ObjectManager::removeBlockTileOverlays() {

    for (auto overlay : blockedTileOverlays_) {
        gameScene_.lock()->removeItem(overlay);
    }

    blockedTileOverlays_.clear();
}


int ObjectManager::getTileCount()
{
    return tiles_.size();
}


int ObjectManager::getTileCountForPlayer
                        (std::shared_ptr<Course::PlayerBase> player)
{
    int tileCount = 0;
    for (auto tile : tiles_) {
        if (tile->getOwner() == player) {
            ++tileCount;
        }
    }
    return tileCount;
}


int ObjectManager::getNeutralTiles()
{
    int tileCount = 0;
    for (auto tile : tiles_) {
        if (tile->getOwner() == nullptr) {
            ++tileCount;
        }
    }
    return tileCount;
}


} //Namespace Student



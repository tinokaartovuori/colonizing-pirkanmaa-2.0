/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: blockedtile.h, header for BlockedTile class                  #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef BLOCKEDTILE_H
#define BLOCKEDTILE_H


#include "Core/gameobject.h"


namespace Student {


/**
 * @brief Tile that blocks the view on which the player cannot place an unit
 */
class BlockedTile : public Course::GameObject
{
public:
    BlockedTile(const Course::Coordinate& coordinate,
             int width,
             int height,
             const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
             const std::weak_ptr<Course::iObjectManager>& objectmanager);

    virtual std::string getType() const override;
};

} //namespace Student


#endif // BLOCKEDTILE_H

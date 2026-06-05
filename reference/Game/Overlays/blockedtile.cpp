/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: blockedtile.cpp, see blockedtile.h for the class's description #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "blockedtile.h"
#include <qdebug.h>


namespace Student {

BlockedTile::BlockedTile(const Course::Coordinate &coordinate,
        int width,
        int height,
        const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
        const std::weak_ptr<Course::iObjectManager> &objectmanager):
    Course::GameObject(coordinate, width, height, eventhandler, objectmanager)
{
}

std::string BlockedTile::getType() const
{
    return "BlockedTile";
}

} //namespace Course

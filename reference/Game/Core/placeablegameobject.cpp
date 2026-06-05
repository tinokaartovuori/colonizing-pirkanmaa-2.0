/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: placeablegameobject.cpp, see placeablegameobject.h for more info #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "placeablegameobject.h"
#include "Tiles/tilebase.h"

#include "Exceptions/ownerconflict.h"
#include <qdebug.h>

namespace Course {

PlaceableGameObject::PlaceableGameObject
       (const std::weak_ptr<iGameEventHandler> &eventhandler,
        const std::weak_ptr<iObjectManager> &objectmanager,
        const std::weak_ptr<PlayerBase> &owner
        ):
    GameObject(owner, eventhandler, objectmanager),
    m_location({})
{
}

std::string PlaceableGameObject::getType() const
{
    return "PlaceableGameObject";
}


bool PlaceableGameObject::canBePlacedOnTile(const std::shared_ptr<TileBase> &target) const
{
    if( target->getOwner() == nullptr or getOwner() == nullptr )
    {
        return true;
    }

    return hasSameOwnerAs(target);
}

void PlaceableGameObject::setLocationTile(const std::shared_ptr<TileBase>& tile)
{
    if( tile )
    {
        if( not canBePlacedOnTile(tile) )
        {
            throw IllegalAction("Illegal action for " + getType());
        }
        setCoordinate(tile->getCoordinate());
        m_location = tile;
    }

}

std::shared_ptr<TileBase> PlaceableGameObject::currentLocationTile() const
{
    return m_location.lock();
}

ResourceMap PlaceableGameObject::getCost()
{
    return {};
}



} // namespace Course

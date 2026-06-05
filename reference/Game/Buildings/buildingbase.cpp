/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: buildingbase.cpp, see buildingbase.h for the class's description   #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "buildingbase.h"
#include "Exceptions/ownerconflict.h"
#include "Tiles/tilebase.h"

#include "qdebug.h"


namespace Course {


BuildingBase::BuildingBase(const std::weak_ptr<iGameEventHandler>& eventhandler,
        const std::weak_ptr<iObjectManager> &objectmanager,
        const std::weak_ptr<PlayerBase>& owner,
        const ResourceMap& buildcost,
        const ResourceMap& production,
        const std::string& basic_description):
    PlaceableGameObject(eventhandler, objectmanager, owner),
    BUILD_COST(buildcost),
    PRODUCTION_EFFECT(production),
    basicDescription_(basic_description)
{
}

std::string BuildingBase::getType() const
{
    return "BuildingBase";
}


ResourceMap BuildingBase::getProduction()
{
    return PRODUCTION_EFFECT;
}

ResourceMap BuildingBase::getCost()
{
    return BUILD_COST;
}

void BuildingBase::addBasicDescription(std::string desc)
{
    basicDescription_ = desc;
}

std::string BuildingBase::getBasicDescription()
{
    return basicDescription_;
}

void BuildingBase::setParentTile(std::shared_ptr<TileBase> parentTile)
{
    parentTile_ = parentTile;
}



} // namespace Course

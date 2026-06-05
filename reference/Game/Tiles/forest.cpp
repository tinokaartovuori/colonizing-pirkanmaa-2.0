/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: forest.cpp, see forest.h for the class's description               #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "forest.h"
#include "grassland.h"


namespace Course {

Forest::Forest(const Coordinate& location,
               int size_x,
               int size_y,
               const std::weak_ptr<iGameEventHandler> &eventhandler,
               const std::weak_ptr<iObjectManager> &objectmanager,
               const unsigned int& max_units,
               const ResourceMap& production):
    TileBase(location, size_x, size_y,
             eventhandler,objectmanager,
             max_units, production,
             Student::ConstDescriptionMaps::FOREST_DESCRIPTION),
             woodLeft_(ConstResourceMaps::FOREST_CAPACITY.at(BasicResource::WOOD)),
             roundsStumpsHaveBeen_(0)
{
}


std::string Forest::getType() const
{
    return "Forest";
}


std::vector<std::string> Forest::getBuildableBuildings()
{
    if (woodLeft_ == 0) {
        //Check if neighbour tiles contains headquartes or outpost
        for (auto tile : getNeighbourTiles()) {
            if (tile->getBuilding() != nullptr) {
                if (tile->getBuilding()->getType() == "Headquarters" or
                        tile->getBuilding()->getType() == "Outpost") {
                    return  {"Farm",
                            "Village",
                            "Nuclear Power Plant"
                             };
                }
            }
        }

        return  {"Farm",
                "Village",
                "Outpost",
                "Nuclear Power Plant"
                 };
    }
    else {
        return {};
    }
}


void Forest::generateResources()
{
    //Production is multiplied by the number of basic workers
    for (auto unit : getUnits()) {
        if (unit->getType() == "BasicWorker" && woodLeft_ > 0) {
            owner_.lock()->addOrRemoveResources
                                (ConstResourceMaps::FOREST_PRODUCTION);
            woodLeft_ -= ConstResourceMaps::FOREST_PRODUCTION.at
                                                    (BasicResource::WOOD);
        }
    }

    std::shared_ptr<Forest> thisTile =
            std::dynamic_pointer_cast<Forest>(shared_from_this());

    //The forest grows back so the image is updated and the wood regenerates
    if (roundsStumpsHaveBeen_ == ConstResourceMaps::FOREST_GROW_TIME) {
        roundsStumpsHaveBeen_ = 0;
        woodLeft_ = ConstResourceMaps::FOREST_CAPACITY.at(BasicResource::WOOD);
        lockEventHandler()->updateForest("Grow", thisTile);
    }
    //The forest has run out of wood so the image is updated
    else if (woodLeft_ == 0 && roundsStumpsHaveBeen_ == 0) {
        lockEventHandler()->updateForest("Cut", thisTile);
        ++roundsStumpsHaveBeen_;
    }
    //Keeps track how many rounds the forest has been empty
    else if (woodLeft_ == 0) {
        ++roundsStumpsHaveBeen_;
    }
}


ResourceMap Forest::getCurrentRevenue()
{
    int wl = woodLeft_;
    Course::ResourceMap production = Course::ConstResourceMaps::NO_RESOURCES;
    for (auto unit : getUnits()) {
        if (unit->getType() == "BasicWorker" && wl > 0) {
            production = mergeResourceMaps(production,
                                             ConstResourceMaps::FOREST_PRODUCTION);
            wl = wl - ConstResourceMaps::FOREST_PRODUCTION.at(BasicResource::WOOD);
        }
    }

    return production;
}


std::string Forest::getExtraDescription()
{
    if (woodLeft_ == ConstResourceMaps::FOREST_CAPACITY.at(BasicResource::WOOD)) {
        return "";
    }
    if (woodLeft_ > 0 and roundsStumpsHaveBeen_ == 0) {

        return "<u>Forest cut:</u><br>" + std::to_string(woodLeft_/
              ConstResourceMaps::FOREST_PRODUCTION.at(BasicResource::WOOD))
                + "/" + std::to_string(ConstResourceMaps::FOREST_CAPACITY.at
               (BasicResource::WOOD)/ConstResourceMaps::FOREST_PRODUCTION.at
               (BasicResource::WOOD));

    }
    if (roundsStumpsHaveBeen_ > 0) {
        return "<u>Forest growth:</u><br>" +
                std::to_string(roundsStumpsHaveBeen_ - 1) + "/" +
                std::to_string(ConstResourceMaps::FOREST_GROW_TIME);
    }

    return "";
}



} // namespace Course

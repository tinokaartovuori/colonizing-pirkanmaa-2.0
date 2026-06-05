/*
##############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                  #
#                                                                            #
# Project: Colonizing Pirkanmaa                                              #
# Program description: Program instructions are located in                   #
#                      Documentation/documentation.pdf                       #
#                                                                            #
# File: AbundantForest.cpp, see AbundantForest.h for the class's description #
#                                                                            #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi                #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi                #
##############################################################################
*/

#include "abundantforest.h"
#include "grassland.h"
#include "Core/resourcemaps.h"


namespace Student {

AbundantForest::AbundantForest(const Course::Coordinate& location,
               int size_x,
               int size_y,
               const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
               const std::weak_ptr<Course::iObjectManager>& objectmanager,
               const unsigned int& max_units,
               const Course::ResourceMap& production):
    TileBase(location, size_x, size_y,
             eventhandler,objectmanager,
             max_units, production,
             Student::ConstDescriptionMaps::ABUNDANT_FOREST_DESCRIPTION)
{
}


std::string AbundantForest::getType() const
{
    return "Abundant Forest";
}


std::vector<std::string> AbundantForest::getBuildableBuildings()
{
    return {};
}


void AbundantForest::generateResources()
{
    //Production is multiplied by the number of basic workers
    for (auto unit : getUnits()) {
        if (unit->getType() == "BasicWorker") {
            owner_.lock()->addOrRemoveResources
                                (Course::ConstResourceMaps::ABUNDANT_FOREST_PRODUCTION);
            break;
        }
    }

}


Course::ResourceMap AbundantForest::getCurrentRevenue()
{
    Course::ResourceMap production = Course::ConstResourceMaps::NO_RESOURCES;
    for (auto unit : getUnits()) {
        if (unit->getType() == "BasicWorker") {
            production = mergeResourceMaps(production,
                                             Course::ConstResourceMaps::ABUNDANT_FOREST_PRODUCTION);
            break;
        }
    }

    return production;
}

std::string AbundantForest::getExtraDescription()
{
    return "";
}


} // namespace Course

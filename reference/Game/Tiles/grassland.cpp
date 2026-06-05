/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: grassland.cpp, see grassland.h for the class's description         #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/


#include "grassland.h"


namespace Course {

Grassland::Grassland(const Coordinate& location,
                     int size_x,
                     int size_y,
                     const std::weak_ptr<iGameEventHandler>& eventhandler,
                     const std::weak_ptr<iObjectManager>& objectmanager,
                     const unsigned int& max_units,
                     const ResourceMap& production):
    TileBase(location,
             size_x,
             size_y,
             eventhandler,
             objectmanager,
             max_units,
             production,
             Student::ConstDescriptionMaps::GRASSLAND_DESCRIPTION)
{
}

std::string Grassland::getType() const
{
    return "Grassland";
}


std::vector<std::string> Grassland::getBuildableBuildings()
{
    //Check if neighbour tiles contains headquartes or outpost
    for (auto tile : getNeighbourTiles()) {
        if (tile->getBuilding() != nullptr) {
            if (tile->getOwner() == getOwner() and (tile->getBuilding()->getType() == "Headquarters" or
                    tile->getBuilding()->getType() == "Outpost")) {
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


void Grassland::generateResources()
{
    std::shared_ptr<Grassland> thisTile =
            std::dynamic_pointer_cast<Grassland>(shared_from_this());

    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Farm") {
            bool hasWorker = false;
            for (auto unit : getUnits()) {
                if (unit->getType() == "BasicWorker") {
                    hasWorker = true;
                }
            }

            int growthPhase = std::static_pointer_cast<Farm>(getBuilding())
                         ->getGrowthPhase() +1;

            std::shared_ptr<Farm> farm =
                                std::static_pointer_cast<Farm>(getBuilding());

            farm->setGrowthPhase(growthPhase);

            if (growthPhase == 5  && hasWorker) {
                owner_.lock()->addOrRemoveResources(getBuilding()
                                                    ->getProduction());
                farm->resetFarm();
            }
            else if (hasWorker) {
                lockEventHandler()->updateAnimatedTileToStatic(thisTile, growthPhase);
            }
            else if (!hasWorker) {
                farm->resetFarm();
            }
        }
        if (getBuilding()->getType() == "Nuclear Power Plant") {
            bool hasExpert = false;
            for (auto unit : getUnits()) {
                if (unit->getType() == "Expert") {
                    hasExpert = true;
                }
            }
            if (!hasExpert) {
                return;
            }
            for (auto unit : getUnits()) {
                if (unit->getType() == "BasicWorker") {
                    owner_.lock()->addOrRemoveResources(getBuilding()
                                                        ->getProduction());
                }
            }
        }
        if (getBuilding()->getType() == "Village") {
            owner_.lock()->addOrRemoveResources(getBuilding()
                                                ->getProduction());
        }
        if (getBuilding()->getType() == "Outpost") {
            owner_.lock()->addOrRemoveResources(getBuilding()
                                                ->getProduction());
        }
    }
}


ResourceMap Grassland::getCurrentRevenue()
{
    ResourceMap production = Course::ConstResourceMaps::NO_RESOURCES;
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Farm") {
            bool hasWorker = false;
            for (auto unit : getUnits()) {
                if (unit->getType() == "BasicWorker") {
                    hasWorker = true;
                }
            }
            int growthPhase = std::static_pointer_cast<Farm>
                    (getBuilding())->getGrowthPhase() + 1;
            if (growthPhase == 5  && hasWorker) {
                production = mergeResourceMaps(production,
                                               getBuilding()->getProduction());
            }
        }
        if (getBuilding()->getType() == "Nuclear Power Plant") {
            bool hasExpert = false;
            for (auto unit : getUnits()) {
                if (unit->getType() == "Expert") {
                    hasExpert = true;
                }
            }
            if (!hasExpert) {
                return production;
            }
            for (auto unit : getUnits()) {
                if (unit->getType() == "BasicWorker") {
                    production = mergeResourceMaps(production,
                                                   getBuilding()->getProduction());
                }
            }
        }

    }

    return production;
}


std::string Grassland::getExtraDescription()
{
    if (getBuilding() != nullptr) {

        if (getBuilding()->getType() == "Farm") {
            if (std::dynamic_pointer_cast<Course::Farm>
                    (getBuilding())->getGrowthPhase() ==
                    Course::ConstResourceMaps::FARM_GROW_TIME)
            {
                return "<u>Growth:</u><br>Ready next round!";
            }
            return "<u>Growth:</u><br>" +
                    std::to_string(std::dynamic_pointer_cast<Course::Farm>
                            (getBuilding())->getGrowthPhase() - 1) + "/" +
                    std::to_string(Course::ConstResourceMaps::FARM_GROW_TIME);
        }
        if (getBuilding()->getType() == "Outpost") {
            return std::dynamic_pointer_cast<Course::Outpost>(getBuilding())->getExtraDescription();
        }
        if (getBuilding()->getType() == "Village") {

            return std::dynamic_pointer_cast<Student::Village>(getBuilding())->getExtraDescription();
        }
        if (getBuilding()->getType() == "Nuclear Power Plant") {
            if (getUnitCount() == 0) {
                return "";
            } else {
                bool expert = false;
                for (auto unit : getUnits()) {
                    if (unit->getType() == "Expert") {
                        expert = true;
                    }
                }
                if (!expert) {
                    return "Expert is missing!";
                } else {
                    return "";
                }
            }
        }
    }

    return "";
}


std::string Grassland::getNetDescription()
{
    if (getBuilding()->getType() == "Headquarters") {
        return std::dynamic_pointer_cast<Course::HeadQuarters>
                (getBuilding())->getExtraDescription();
    }

    if (getBuilding()->getType() == "Mikontalo") {
        return std::dynamic_pointer_cast<Student::Mikontalo>
                (getBuilding())->getExtraDescription();
    }

    std::string functionalString = "<u>Net value:</u>";

    if (getCurrentNet() == Course::ConstResourceMaps::NO_RESOURCES
          || getCurrentNet() == Course::ConstResourceMaps::EMPTY)
    {
        if (getBuilding()->getType() == "Farm") {
            return functionalString + "<br>No money this round.";
        } else {
            return "";
        }
    }

    for (auto res : getCurrentNet()) {

        if (res.first == Course::BasicResource::MONEY && res.second != 0) {
            functionalString += "<br>";
            functionalString += std::to_string(res.second) + " Money/r";
        }
        if (res.first == Course::BasicResource::WOOD && res.second != 0) {
            functionalString += "<br>";
            functionalString += std::to_string(res.second) + " Wood/r";
        }
        if (res.first == Course::BasicResource::STONE && res.second != 0) {
            functionalString += "<br>";
            functionalString += std::to_string(res.second) + " Stone/r";
        }
        if (res.first == Course::BasicResource::METAL && res.second != 0) {
            functionalString += "<br>";
            functionalString += std::to_string(res.second) + " Metal/r";
        }

    }

    return functionalString;
}


bool Grassland::hasOpponentHeadquarters(std::shared_ptr<PlayerBase> player)
{
    //Player cannot place an unit on his own headquarters, only opponents
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Headquarters"
                       && getOwner() == player)
        {
            std::shared_ptr<HeadQuarters> HQ =
                     std::dynamic_pointer_cast<HeadQuarters>(getBuilding());

            if (!HQ->isConquered()) {
                return false;
            }
        }
    }

    return true;
}


int Grassland::getMaxUnitsIncrease()
{
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Headquarters") {
            if (!std::dynamic_pointer_cast<Course::HeadQuarters>(getBuilding())->isConquered()) {
                return Course::ConstResourceMaps::HQ_UNIT_VALUE;
            }
        }
        if (getBuilding()->getType() == "Village") {
            return Course::ConstResourceMaps::VILLAGE_UNIT_VALUE;
        }
        if (getBuilding()->getType() == "Mikontalo") {
            return Course::ConstResourceMaps::MIKONTALO_UNIT_VALUE;
        }
    }

    return 0;
}


int Grassland::getMaxSoldiersIncrease()
{
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Headquarters") {
            if (!std::dynamic_pointer_cast<Course::HeadQuarters>(getBuilding())->isConquered()) {
                return Course::ConstResourceMaps::HQ_SOLDIER_VALUE;
            }
        }
        if (getBuilding()->getType() == "Outpost") {
            return Course::ConstResourceMaps::OUTPOST_SOLDIER_VALUE;
        }
    }

    return 0;
}


void Grassland::updateAnimation()
{
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Nuclear Power Plant" and
                getCurrentRevenue().at(Course::BasicResource::MONEY) > 0)
        {
            getBuilding()->setAnimationOption(AnimationOptions::NUCLEAR);
            lockObjectManager()->getGameScene()->getObjectInScene(getBuilding())
                    ->setAnimationOption(AnimationOptions::NUCLEAR);
            lockObjectManager()->getGameScene()->getObjectInScene(getBuilding())
                    ->setAnimationFrame(1);

        } else if (getBuilding()->getType() == "Nuclear Power Plant")
        {
            getBuilding()->setAnimationOption(AnimationOptions::EMPTY);
            lockObjectManager()->getGameScene()->getObjectInScene(getBuilding())
                    ->setAnimationOption(AnimationOptions::EMPTY);
            lockObjectManager()->getGameScene()->getObjectInScene(getBuilding())
                    ->setAnimationFrame(1);
        }
    }
}



} // namespace Course

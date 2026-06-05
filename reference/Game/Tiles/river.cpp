/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: river.cpp, see river.h for the class's description                 #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/


#include "river.h"

#include "Exceptions/notenoughspace.h"
#include "Exceptions/ownerconflict.h"
#include "Exceptions/invalidpointer.h"
#include "Graphics/sceneitem.h"

namespace Student {

River::River(const Course::Coordinate& location,
                     int size_x,
                     int size_y,
                     const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
                     const std::weak_ptr<Course::iObjectManager> &objectmanager,
                     const unsigned int& max_units,
                     const Course::ResourceMap& production):
    TileBase(location,
             size_x,
             size_y,
             eventhandler,
             objectmanager,
             max_units,
             production)
{
    riverOrientation_ = 3;
    riverShape_ = "";
}


std::string River::getType() const
{
    return "River";
}


std::vector<std::string> River::getBuildableBuildings()
{
    if (riverOrientation_ == 1 or riverOrientation_ == 0) {
        return  {"Bridge", "Hydroelectric Power Plant"};
    }
    else {
        return {};
    }
}


void River::generateResources()
{
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Hydroelectric Power Plant") {
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

        if (getBuilding()->getType() == "Bridge") {
            owner_.lock()->addOrRemoveResources(getBuilding()
                                                ->getProduction());
        }
    }
}


Course::ResourceMap River::getCurrentRevenue() {

    Course::ResourceMap production = Course::ConstResourceMaps::NO_RESOURCES;
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Hydroelectric Power Plant") {
            bool hasExpert = false;
            for (auto unit : getUnits()) {
                if (unit->getType() == "Expert") {
                    hasExpert = true;
                }
            }
            if (!hasExpert) {
                return Course::ConstResourceMaps::NO_RESOURCES;
            }
            for (auto unit : getUnits()) {
                if (unit->getType() == "BasicWorker") {
                     production = mergeResourceMaps(production, getPositivesMap(getBuilding()->getProduction()));

                }
            }
            return production;
        }

        if (getBuilding()->getType() == "Bridge") {
            return mergeResourceMaps(production, getPositivesMap(getBuilding()->getProduction()));
        }
    }
    return Course::ConstResourceMaps::NO_RESOURCES;
}


std::string River::getExtraDescription()
{
    if (getBuilding() != nullptr) {
        if (getBuilding()->getType() == "Hydroelectric Power Plant") {
            if (getUnitCount() != 0) {
                bool hasExpert = false;
                for (auto unit : getUnits()) {
                    if (unit->getType() == "Expert") {
                        hasExpert = true;
                    }
                }
                if (!hasExpert) {
                    return "Expert is missing!";
                }
            }
        }
    }

    return "";
}


void River::addUnit(const std::shared_ptr<Course::UnitBase>& unit)
{
    std::shared_ptr<TileBase> thisTile =
            std::dynamic_pointer_cast<River>(shared_from_this());

    if (unit->getType() == "BasicWorker") {
        unit->setImageFiles(ImageVectors::BASICWORKER);
    }
    if (unit->getType() == "Expert") {
        unit->setImageFiles(ImageVectors::EXPERT);
    }
    if (unit->getType() == "Soldier") {
        unit->setImageFiles(ImageVectors::SOLDIER);
    }

    if (not thisTile)
    {
        throw Course::InvalidPointer("Objectmanager didn't find Tile: " +
                             std::to_string(ID));
    }
    if (!unit->isConqueringUnit()) {
        if (getUnitCount() + 1 > 3)
        {
            throw Course::NotEnoughSpace("Tile: " + std::to_string(ID) +
                                 " has no more room for Units!");
        }
        unit->setLocationTile(thisTile);
        int index = getUnitCount();
        bool changeImage = false;

        if (riverShape_ == "NS" or riverShape_ == "SE" or riverShape_ == "SW") {
            if (index == 1) {
                changeImage = true;
            }
        }

        if (changeImage and getBuilding() == nullptr) {
            if (unit->getType() == "BasicWorker") {
                unit->setImageFiles(ImageVectors::BASICWORKER_SWIM);
            }
            if (unit->getType() == "Expert") {
                unit->setImageFiles(ImageVectors::EXPERT_SWIM);
            }
            if (unit->getType() == "Soldier") {
                unit->setImageFiles(ImageVectors::SOLDIER_SWIM);
            }

        }

        units_.push_back(unit);
    }
    else {
        if (getConqueringUnitCount() + 1 > 3)
        {
            throw Course::NotEnoughSpace("Tile: " + std::to_string(ID) +
                                 " has no more room for Enemy units!");
        }
        unit->setLocationTile(thisTile);

        int index = getConqueringUnitCount();
        bool changeImage = false;
        if (index == 1) {
            changeImage = true;
        }

        if (riverShape_ == "EW") {
            changeImage = true;
        }
        if (riverShape_ == "SE" or riverShape_ == "NE") {
            if (index == 2) {
                changeImage = true;
            }
        }
        if (riverShape_ == "NW" or riverShape_ == "SW") {
            if (index == 0) {
                changeImage = true;
            }
        }

        if (changeImage and getBuilding() == nullptr) {
            if (unit->getType() == "BasicWorker") {
                unit->setImageFiles(ImageVectors::BASICWORKER_SWIM);
            }
            if (unit->getType() == "Expert") {
                unit->setImageFiles(ImageVectors::EXPERT_SWIM);
            }
            if (unit->getType() == "Soldier") {
                unit->setImageFiles(ImageVectors::SOLDIER_SWIM);
            }

        }

        conqueringUnits_.push_back(unit);
    }

}


void River::updateUnitCoordinates() {
    int ind = 0;
    if (units_.size() > 0)  {
        for(std::vector<std::shared_ptr<Course::UnitBase>>::iterator it = units_.begin(); it != units_.end(); ++it)
        {
            if ((*it)->getType() == "BasicWorker") {
                (*it)->setImageFiles(ImageVectors::BASICWORKER);
            }
            if ((*it)->getType() == "Expert") {
                (*it)->setImageFiles(ImageVectors::EXPERT);
            }
            if ((*it)->getType() == "Soldier") {
                (*it)->setImageFiles(ImageVectors::SOLDIER);
            }

            (*it)->setTileRelatedCoordinates(ind, 1);

            bool changeImage = false;
            if (riverShape_ == "NS" or riverShape_ == "SE" or riverShape_ == "SW") {
                if (ind == 1) {
                    changeImage = true;
                }
            }
            if (changeImage and getBuilding() == nullptr) {
                if ((*it)->getType() == "BasicWorker") {
                    (*it)->setImageFiles(ImageVectors::BASICWORKER_SWIM);
                }
                if ((*it)->getType() == "Expert") {
                    (*it)->setImageFiles(ImageVectors::EXPERT_SWIM);
                }
                if ((*it)->getType() == "Soldier") {
                    (*it)->setImageFiles(ImageVectors::SOLDIER_SWIM);
                }

            }

            ++ind;
        }
    }
    ind = 0;

    if (conqueringUnits_.size() > 0)  {
        for(std::vector<std::shared_ptr<Course::UnitBase>>::iterator it = conqueringUnits_.begin(); it != conqueringUnits_.end(); ++it)
        {
            if ((*it)->getType() == "BasicWorker") {
                (*it)->setImageFiles(ImageVectors::BASICWORKER);
            }
            if ((*it)->getType() == "Expert") {
                (*it)->setImageFiles(ImageVectors::EXPERT);
            }
            if ((*it)->getType() == "Soldier") {
                (*it)->setImageFiles(ImageVectors::SOLDIER);
            }

            (*it)->setTileRelatedCoordinates(ind, 0);


            bool changeImage = false;
            if (ind == 1) {
                changeImage = true;
            }

            if (riverShape_ == "EW") {
                changeImage = true;
            }
            if (riverShape_ == "SE" or riverShape_ == "NE") {
                if (ind == 2) {
                    changeImage = true;
                }
            }
            if (riverShape_ == "NW" or riverShape_ == "SW") {
                if (ind == 0) {
                    changeImage = true;
                }
            }
            if (changeImage and getBuilding() == nullptr) {
                if ((*it)->getType() == "BasicWorker") {
                    (*it)->setImageFiles(ImageVectors::BASICWORKER_SWIM);
                }
                if ((*it)->getType() == "Expert") {
                    (*it)->setImageFiles(ImageVectors::EXPERT_SWIM);
                }
                if ((*it)->getType() == "Soldier") {
                    (*it)->setImageFiles(ImageVectors::SOLDIER_SWIM);
                }

            }

            ++ind;
        }
    }
}


void River::updateAnimation() {
    if (getBuilding() != nullptr ) {
        if (getBuilding()->getType() == "Hydroelectric Power Plant" and
                getCurrentRevenue().at(Course::BasicResource::MONEY) > 0) {
            getBuilding()->setAnimationOption(AnimationOptions::HEPP);
            lockObjectManager()->getGameScene()->getObjectInScene(getBuilding())
                    ->setAnimationOption(AnimationOptions::HEPP);
            lockObjectManager()->getGameScene()->getObjectInScene(getBuilding())
                    ->setAnimationFrame(1);
        } else {
            getBuilding()->setAnimationOption(AnimationOptions::EMPTY);
            lockObjectManager()->getGameScene()->getObjectInScene(getBuilding())
                    ->setAnimationOption(AnimationOptions::EMPTY);
            lockObjectManager()->getGameScene()->getObjectInScene(getBuilding())
                    ->setAnimationFrame(1);
        }
    }
}


int River::getRiverOrientation()
{
    return riverOrientation_;
}


void River::setRiverOrientation(int ori) {
    riverOrientation_ = ori;
    if (ori == 3) {
        addBasicDescription(Student::ConstDescriptionMaps::RIVER_DESCRIPTION_2);
    } else {
        addBasicDescription(Student::ConstDescriptionMaps::RIVER_DESCRIPTION_1);
    }
}


std::string River::getRiverShape()
{
    return riverShape_;
}


void River::setRiverShape(std::string shape) {
    riverShape_ = shape;
}


} // namespace Student


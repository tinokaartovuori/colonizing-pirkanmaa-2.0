/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: unitbase.cpp, see unitbase.h for the class's description           #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "unitbase.h"
#include "Tiles/tilebase.h"
#include <QDebug>


namespace Course {

UnitBase::UnitBase(const std::weak_ptr<iGameEventHandler>& eventhandler,
         const std::weak_ptr<iObjectManager>& objectmanager,
         const std::weak_ptr<Student::GameSettingsManager>& gamesettingsmanager,
         const std::weak_ptr<PlayerBase>& owner,
         const std::weak_ptr<TileBase>& parenttile):
         PlaceableGameObject(eventhandler, objectmanager, owner),
         gameSettingsManager_(gamesettingsmanager),
         parentTile_(parenttile)
{
    if (parentTile_.lock()->getOwner() != getOwner()) {
        tileRelativeCoordinate_ = std::make_shared<Course::Coordinate>
                                        (parentTile_.lock()->getUnitCount(), 1);
        isConqueringUnit_ = false;
    } else {
        tileRelativeCoordinate_ = std::make_shared<Course::Coordinate>
                               (parentTile_.lock()->getConqueringUnitCount(), 0);
        isConqueringUnit_ = true;
    }
}


UnitBase::UnitBase(const std::weak_ptr<iGameEventHandler>& eventhandler,
         const std::weak_ptr<iObjectManager>& objectmanager,
         const std::weak_ptr<Student::GameSettingsManager>& gamesettingsmanager,
         const std::weak_ptr<PlayerBase>& owner):
         PlaceableGameObject(eventhandler, objectmanager, owner),
         gameSettingsManager_(gamesettingsmanager),
         parentTile_()
{
}


std::string UnitBase::getType() const
{
    return "UnitBase";
}


void UnitBase::addParentTile(std::shared_ptr<Course::TileBase> tile)
{
    parentTile_ = tile;
    if (parentTile_.lock()->getOwner() == getOwner()) {
        tileRelativeCoordinate_ = std::make_shared<Course::Coordinate>
                             (parentTile_.lock()->getUnitCount(), 1);
        isConqueringUnit_ = false;
    } else {
        tileRelativeCoordinate_ = std::make_shared<Course::Coordinate>
                             (parentTile_.lock()->getConqueringUnitCount(), 0);
        isConqueringUnit_ = true;
    }
}


std::shared_ptr<Course::TileBase> UnitBase::getParentTile()
{
    return parentTile_.lock();
}


void UnitBase::updateParentTile()
{
    if (parentTile_.lock()->getOwner() == getOwner()) {
        setTileRelatedCoordinates(parentTile_.lock()->getUnitCount(), 1);
        isConqueringUnit_ = false;
    }
    else {
        setTileRelatedCoordinates(parentTile_.lock()->getUnitCount(), 1);
        isConqueringUnit_ = true;
    }
}


bool UnitBase::canBePlacedOnTile(const std::shared_ptr<TileBase> &target) const
{
    //Checks if the tile has space for the unit
    if ((isConqueringUnit_ && target->hasSpaceForConqueringUnits()) ||
            (!isConqueringUnit_ && target->hasSpaceForUnits()))
    {
        std::vector<std::shared_ptr<Course::TileBase>> availableTiles =
                                        lockObjectManager()->getAvailableTiles();
        for (auto tile : availableTiles) {
            if (tile == target) {
                return true;
            }
        }

        return false;
    }

    return false;
}


int UnitBase::getGridSize()
{
    return gameSettingsManager_.lock()->getMapGridSize();
}


std::shared_ptr<Course::Coordinate> UnitBase::getTileRelatedCoordinates()
{
    return tileRelativeCoordinate_;
}


void UnitBase::setTileRelatedCoordinates(int x, int y)
{
    tileRelativeCoordinate_->set_x(x);
    tileRelativeCoordinate_->set_y(y);
}


void UnitBase::paySalary()
{
    owner_.lock()->addOrRemoveResources(getSalary());
}


bool UnitBase::isConqueringUnit() {
    return isConqueringUnit_;
}


void UnitBase::setAsConquering(bool isConquering)
{
    isConqueringUnit_ = isConquering;
}



} // namespace Course

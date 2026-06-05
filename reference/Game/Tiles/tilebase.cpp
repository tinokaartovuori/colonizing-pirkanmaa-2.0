/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: tilebase.cpp, see tilebase.h for the class's description           #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/


#include "tilebase.h"

#include <QtGlobal> // For Q_ASSERT
#include <iostream>
#include <QDebug>

#include "Exceptions/notenoughspace.h"
#include "Exceptions/ownerconflict.h"
#include "Exceptions/invalidpointer.h"



namespace Course {

TileBase::TileBase(const Coordinate& location,
                   int size_x,
                   int size_y,
                   const std::weak_ptr<iGameEventHandler> &eventhandler,
                   const std::weak_ptr<iObjectManager> &objectmanager,
                   const unsigned int& max_units,
                   const ResourceMap& production,
                   const std::string basic_description):
    GameObject(location, size_x, size_y, eventhandler, objectmanager),
    MAX_UNITS(max_units),
    BASE_PRODUCTION(production),
    basicDescription_(basic_description),
    units_({}),
    conqueringUnits_({})
{
}


TileBase::TileBase(const Coordinate &location,
                   int size_x,
                   int size_y,
                   const std::weak_ptr<iGameEventHandler> &eventhandler,
                   const std::weak_ptr<iObjectManager> &objectmanager,
                   const unsigned int &max_units,
                   const ResourceMap &production):
    GameObject(location, size_x, size_y, eventhandler, objectmanager),
    MAX_UNITS(max_units),
    BASE_PRODUCTION(production)
{
}


std::string TileBase::getType() const
{
    return "TileBase";
}


void TileBase::addUnit(const std::shared_ptr<UnitBase>& unit)
{

    if (unit->getType() == "BasicWorker") {
        unit->setImageFiles(ImageVectors::BASICWORKER);
    }
    if (unit->getType() == "Expert") {
        unit->setImageFiles(ImageVectors::EXPERT);
    }
    if (unit->getType() == "Soldier") {
        unit->setImageFiles(ImageVectors::SOLDIER);
    }

    std::shared_ptr<TileBase> thisTile =
            std::dynamic_pointer_cast<TileBase>(shared_from_this());
    if (not thisTile)
    {
        throw InvalidPointer("Objectmanager didn't find Tile: " +
                             std::to_string(ID));
    }
    if (!unit->isConqueringUnit()) {
        if (getUnitCount() + 1 > MAX_UNITS)
        {
            throw NotEnoughSpace("Tile: " + std::to_string(ID) +
                                 " has no more room for Units!");
        }
        unit->setLocationTile(thisTile);
        units_.push_back(unit);
    }
    else {
        if (getConqueringUnitCount() + 1 > MAX_UNITS)
        {
            throw NotEnoughSpace("Tile: " + std::to_string(ID) +
                                 " has no more room for conquering units!");
        }

        unit->setLocationTile(thisTile);
        conqueringUnits_.push_back(unit);
    }
}


void TileBase::removeUnit(const std::shared_ptr<UnitBase>& unit)
{

    std::shared_ptr<TileBase> thisTile =
                std::dynamic_pointer_cast<TileBase>(shared_from_this());

    /*Checks if the player's unit is in a tile the player owns
     *and removes it if it is found */
    for(std::vector<std::shared_ptr<Course::UnitBase>>::iterator it =
        units_.begin(); it != units_.end();)
    {
        if ((*it) == unit){
            it = units_.erase(it);
            break;
        } else {
            ++it;
        }
    }

    /*Checks if the player's unit is in a tile the player doesn't
     *own (it's conquering) and removes it if it is found */
    for(std::vector<std::shared_ptr<Course::UnitBase>>::iterator it =
        conqueringUnits_.begin(); it != conqueringUnits_.end();)
    {
        if ((*it) == unit){
            it = conqueringUnits_.erase(it);
            break;
        } else {
            ++it;
        }
    }

    updateUnitCoordinates(); //Unit's tile related coordinates are updated
    lockEventHandler()->updateTile(thisTile); //Updated visually on the gamescene
}


void TileBase::addBuilding(const std::shared_ptr<BuildingBase>& building)
{
    std::shared_ptr<TileBase> thisTile =
            std::dynamic_pointer_cast<TileBase>(shared_from_this());

    if (thisTile->getType() == "Forest") {
        lockEventHandler()->updateForest("Grassland", thisTile, building);
        return;
    }

    building->setParentTile(thisTile);
    building->setLocationTile(thisTile);

    building_ = building;


    updateUnitCoordinates();   
}


std::shared_ptr<BuildingBase> TileBase::getBuilding() const
{
    return building_;
}


void TileBase::conquerTile(std::shared_ptr<PlayerBase> currentPlayer)
{
    std::shared_ptr<TileBase> thisTile =
            std::dynamic_pointer_cast<TileBase>(shared_from_this());


    /*If no one owns the tile that has the player's unit on,
     *the player gets the tile*/
    for (auto unit : getConqueringUnits()) {
        if (unit->getOwner() == currentPlayer && getOwner() == nullptr)
        {
            setOwner(currentPlayer);
            unit->setAsConquering(false);

            //Changes the player's units from conquering type into normal type
            for (auto unit : conqueringUnits_) {
                unit->setAsConquering(false);
                units_.push_back(unit);
            }

            conqueringUnits_={};
        }
    }

    /*If someone else owns the tile, the player conquers the tile if he has
     *more soldiers on the tile than the opponent*/
    if (getOwner() != currentPlayer && getOwner() != nullptr) {
        unsigned int ownSoldiers = getSoldierCount();
        unsigned int opponentSoldiers = getOpponentSoldierCount();

        //If tile contains outpost it always wins the conquering
        bool hasOutpost = false;
        if (getBuilding() != nullptr) {
            if (getBuilding()->getType() == "Outpost") {
                hasOutpost = true;
            }
        }

        if (ownSoldiers > opponentSoldiers and !hasOutpost) {
            setOwner(currentPlayer);

            //Updates headquarters image if it is conquered
            if (getBuilding() != nullptr &&
                        getBuilding()->getType() == "Headquarters")
            {
                std::shared_ptr<HeadQuarters> HQ =
                         std::dynamic_pointer_cast<HeadQuarters>(getBuilding());
                HQ->setConquered();
            }

            //Removes opponent's units
            std::vector<std::shared_ptr<UnitBase>> units=units_;
            for (auto unit : units) {
                lockEventHandler()->deleteUnitFromTile(unit, thisTile);
            }

            //Changes the player's units from conquering type into normal type
            for (auto unit : conqueringUnits_) {
                unit->setAsConquering(false);
                units_.push_back(unit);
            }

            conqueringUnits_={};
        }
        else {
            //The player doesn't have enough soldiers so all units will dissappear.
            std::vector<std::shared_ptr<UnitBase>> units=conqueringUnits_;
            for (auto unit : units){
                lockEventHandler()->deleteUnitFromTile(unit, thisTile);
            }
        }
    }

    updateUnitCoordinates();
    lockEventHandler()->updateTile(thisTile);
}


bool TileBase::hasOpponentHeadquarters(std::shared_ptr<PlayerBase> player)
{
    (void)player; //Removes compiler warning
    return true;
}


int TileBase::getMaxUnitsIncrease()
{
    return 0;
}


int TileBase::getMaxSoldiersIncrease()
{
    return 0;
}


unsigned int TileBase::getUnitCount() const
{
    return units_.size();
}


unsigned int TileBase::getConqueringUnitCount() const
{
    return conqueringUnits_.size();
}


unsigned int TileBase::getSoldierCount() const
{
    unsigned int ownSoldiers = 0;
    for (auto unit : getConqueringUnits()) {
        if (unit->getType() == "Soldier") {
            ++ownSoldiers;
        }
    }

    return ownSoldiers;
}


unsigned int TileBase::getOpponentSoldierCount() const
{
    unsigned int opponentSoldiers = 0;
    for (auto unit : getUnits()) {
        if (unit->getType() == "Soldier") {
            ++opponentSoldiers;
        }
    }

    return opponentSoldiers;
}


std::vector< std::shared_ptr<UnitBase> > TileBase::getUnits() const
{
    return units_;
}


std::vector< std::shared_ptr<UnitBase> > TileBase::getConqueringUnits() const
{
    return conqueringUnits_;
}


void TileBase::updateAnimation()
{
    return;
}


ResourceMap TileBase::getCurrentExpenses()
{

    ResourceMap expenses = Course::ConstResourceMaps::NO_RESOURCES;
    for (auto unit : getUnits()) {
        expenses = mergeResourceMaps(expenses, unit->getSalary());
    }
    expenses = getNegativesMap(expenses);
    if (getBuilding() != nullptr) {
        expenses = mergeResourceMaps(expenses,
                                getNegativesMap(getBuilding()->getProduction()));
    }

    return expenses;
}


ResourceMap TileBase::getCurrentNet()
{
    ResourceMap net = Course::ConstResourceMaps::NO_RESOURCES;
    ResourceMap revenue = getCurrentRevenue();
    ResourceMap expenses = getCurrentExpenses();
    net = mergeResourceMaps(revenue, expenses);

    return net;
}


void TileBase::updateUnitCoordinates() {
    int ind = 0;

    if (units_.size() > 0)  {
        for(std::vector<std::shared_ptr<Course::UnitBase>>::iterator it =
                                       units_.begin(); it != units_.end(); ++it)
        {
            (*it)->setTileRelatedCoordinates(ind, 1);
            ++ind;
        }
    }
    ind = 0;
    if (conqueringUnits_.size() > 0)  {
        for(std::vector<std::shared_ptr<Course::UnitBase>>::iterator it =
                   conqueringUnits_.begin(); it != conqueringUnits_.end(); ++it)
        {
            (*it)->setTileRelatedCoordinates(ind, 0);
            ++ind;
        }
    }
}


std::vector<std::shared_ptr<TileBase>> TileBase::getNeighbourFourTiles()
{
    std::shared_ptr<TileBase> thisTile =
            std::dynamic_pointer_cast<TileBase>(shared_from_this());

    std::shared_ptr<Coordinate> coordinates = thisTile->getCoordinatePtr();
    int width = gameSettingsManager_.lock()->getMapGridWidth(); //Map grid width
    int height = gameSettingsManager_.lock()->getMapGridHeight(); //Map grid height

    std::vector<std::shared_ptr<TileBase>> neighbouringTiles;

    //Loops all neighbouring tiles' coordinates
    for (auto neighbourCoordinates : coordinates->neighbouringFour(width, height))
    {
        //Converts coordinates into tiles and adds them into the vector
        std::shared_ptr<TileBase> tile =
                            lockObjectManager()->getTile(neighbourCoordinates);
        neighbouringTiles.push_back(tile);
    }

    return neighbouringTiles;
}

std::vector<std::shared_ptr<TileBase>> TileBase::getNeighbourTiles()
{
    std::shared_ptr<TileBase> thisTile =
            std::dynamic_pointer_cast<TileBase>(shared_from_this());

    std::shared_ptr<Coordinate> coordinates = thisTile->getCoordinatePtr();
    int width = gameSettingsManager_.lock()->getMapGridWidth(); //Map grid width
    int height = gameSettingsManager_.lock()->getMapGridHeight(); //Map grid height

    std::vector<std::shared_ptr<TileBase>> neighbouringTiles;

    //Loops all neighbouring tiles' coordinates
    for (auto neighbourCoordinates : coordinates->neighbours(1, width, height))
    {
        //Converts coordinates into tiles and adds them into the vector
        std::shared_ptr<TileBase> tile =
                            lockObjectManager()->getTile(neighbourCoordinates);
        neighbouringTiles.push_back(tile);
    }

    return neighbouringTiles;
}

bool TileBase::hasSpaceForUnits() const
{
    return 1 + getUnitCount() <= MAX_UNITS;
}


bool TileBase::hasSpaceForConqueringUnits() const
{
    return 1 + getConqueringUnitCount() <= MAX_UNITS;
}


void TileBase::setGameSettings
                     (const std::shared_ptr<Student::GameSettingsManager> manager)
{
    gameSettingsManager_ = manager;
}


void TileBase::clickAction()
{
    std::shared_ptr<TileBase> thisTile =
            std::dynamic_pointer_cast<TileBase>(shared_from_this());

    lockEventHandler()->tileClicked(thisTile);
}


std::vector<QPixmap> TileBase::getOwnerBorderPixmap()
{

    std::shared_ptr<Course::TileBase> westTile = nullptr;
    std::shared_ptr<Course::TileBase> northTile = nullptr;
    std::shared_ptr<Course::TileBase> eastTile = nullptr;
    std::shared_ptr<Course::TileBase> southTile = nullptr;

    //Gets the neighbouring tiles in four different compass directions if they exist
    if (getCoordinatePtr()->x() != 0)
    {
        westTile = lockObjectManager()->getTile
                   (getCoordinatePtr()->neighbour_at(Course::Direction::W, 1));
    }
    if (getCoordinatePtr()->y() != 0)
    {
        northTile = lockObjectManager()->getTile
                    (getCoordinatePtr()->neighbour_at(Course::Direction::N, 1));
    }
    if (getCoordinatePtr()->x() !=
            gameSettingsManager_.lock()->getMapGridWidth() - 1)
    {
        eastTile = lockObjectManager()->getTile
                   (getCoordinatePtr()->neighbour_at(Course::Direction::E, 1));
    }
    if (getCoordinatePtr()->y() !=
            gameSettingsManager_.lock()->getMapGridHeight() - 1)
    {
        southTile = lockObjectManager()->getTile
                    (getCoordinatePtr()->neighbour_at(Course::Direction::S, 1));
    }

    //Tile has an owner so there might be a need to draw the border
    if (getOwner() != nullptr) {
        int playerNum = getOwner()->getPlayerNum();
        std::vector<QPixmap> pixMaps;
        std::vector<std::string> imageNames = ImageVectors::TILEOWNERBORDERS;
        std::string imageName = "";

        //Gets the right colored border image name according to the player number
        if (playerNum == 1) {
            imageName = imageNames.at(0);
        }
        else if (playerNum == 2) {
            imageName = imageNames.at(1);
        }
        else if (playerNum == 3) {
            imageName = imageNames.at(2);
        }
        else if (playerNum == 4) {
            imageName = imageNames.at(3);
        }

        //Creates a QPixmap of the border image
        QString image =
                QString::fromStdString(imageName);
        QPixmap pixItem(image);
        QTransform transform;

        /*Rotates the pixmap according to the direction the border is facing to.
         *The default direction is north.*/
        if (northTile == nullptr || northTile->getOwner() != getOwner())
        {
            pixMaps.push_back(pixItem);
        }
        if (eastTile == nullptr || eastTile->getOwner() != getOwner())
        {
            pixMaps.push_back(pixItem.transformed(transform.rotate(90)));
            pixItem.transformed(transform.rotate(270));
        }
        if (southTile == nullptr || southTile->getOwner() != getOwner())
        {
            pixMaps.push_back(pixItem.transformed(transform.rotate(180)));
            pixItem.transformed(transform.rotate(180));
        }
        if (westTile == nullptr || westTile->getOwner() != getOwner()) {
            pixMaps.push_back(pixItem.transformed(transform.rotate(270)));
            pixItem.transformed(transform.rotate(270));
        }

        //Returns all the pixmaps of the borders that need to be drawn
        if (pixMaps.size() >= 1) {
            return pixMaps;
        }

    }

    return {}; //No need to draw a border so no pixmaps are returned
}


void TileBase::addBasicDescription(std::string desc)
{
    basicDescription_ = desc;
}


std::string TileBase::getBasicDescription()
{
    return basicDescription_;
}


std::string TileBase::getNetDescription()
{
    std::string functionalString = "<u>Net value:</u>";

    if (getCurrentNet() == Course::ConstResourceMaps::NO_RESOURCES
          || getCurrentNet() == Course::ConstResourceMaps::EMPTY)
    {
        return "";
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




} // namespace Course

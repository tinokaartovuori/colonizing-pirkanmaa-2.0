/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: unitbase.h, header for UnitBase-class                        #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef WORKERBASE_H
#define WORKERBASE_H

#include <QDebug>

#include "Core/placeablegameobject.h"
#include "Core/resourcemaps.h"
#include "DAL/gamesettingsmanager.h"


namespace Course {

/**
 * @brief The UnitBase class is aa base-class for Unit-objects.
 *
 * Units can be placed on tiles and they can work in buildings.
 * They can also conquer new areas for the 
 */

class UnitBase : public PlaceableGameObject
{
public:

    /**
     * @brief Disabled parameterless constructor.
     */
    UnitBase() = delete;


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param gamesettingsmanager points to the GameSettingsManager.
     * @param owner is a shared pointer to the owner of the unit
     * @param parenttile is a shared pointer to the tile the unit is on
     */
    UnitBase(const std::weak_ptr<iGameEventHandler> &eventhandler,
        const std::weak_ptr<iObjectManager> &objectmanager,
        const std::weak_ptr<Student::GameSettingsManager> &gamesettingsmanager,
        const std::weak_ptr<PlayerBase> &owner,
        const std::weak_ptr<TileBase> &parenttile);


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param gamesettingsmanager points to the GameSettingsManager.
     * @param owner is a shared pointer to the owner of the unit
     */
    UnitBase(const std::weak_ptr<iGameEventHandler> &eventhandler,
        const std::weak_ptr<iObjectManager> &objectmanager,
        const std::weak_ptr<Student::GameSettingsManager> &gamesettingsmanager,
        const std::weak_ptr<PlayerBase> &owner);


    /**
     * @brief Default destructor.
     */
    virtual ~UnitBase() = default;


    /**
     * @brief Returns the unit's type in string. In this case it's "UnitBase"
     * @return Unit's type in string. In this case it's "UnitBase"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief Adds the parent tile the unit is on.
     * @param Shared pointer to the tile the unit is on.
     * @post Exception guarantee: ?????????????
     */
    void addParentTile(std::shared_ptr<Course::TileBase> tile);


    /**
     * @brief Gets the parent tile the unit is on.
     * @return Shared pointer to the tile the unit is on.
     * @post Exception guarantee: No-throw
     */
    std::shared_ptr<Course::TileBase> getParentTile();


    /**
     * @brief Updates the parent tile the unit is on. The point of the function
     *        is to set the unit tile related coordinates right depending on
     *        if the unit is conquering or not.
     * @post Exception guarantee: ?????????????
     */
    void updateParentTile();


    /**
     * @brief Checks if the unit can be placed on a tile
     * @param Shared pointer of the tile the unit is trying to be placed on
     * @return Boolean value. True if the unit can be placed and false if not
     * @post Exception guarantee: ?????????????
     */
    virtual bool canBePlacedOnTile
                         (const std::shared_ptr<TileBase> &target) const override;


    /**
     * @brief Checks if the unit can be placed on a tile
     * @return Integer of the map grid size (how many pixels wide one map tile is)
     * @post Exception guarantee: ?????????????
     */
    int getGridSize();


    /**
     * @brief Returns the parent tile coordinates
     * @return Shared pointer to the parent tile coordinates
     * @post Exception guarantee: ?????????????
     */
    std::shared_ptr<Coordinate> getTileRelatedCoordinates();


    /**
     * @brief Sets unit's tile related coordinates. Every tile
     *        has a 3x2 grid for units
     * @param Integer of the coordinate x and y values to be set
     * @post Exception guarantee: ?????????????
     */
    void setTileRelatedCoordinates(int x, int y);


    /**
     * @brief Pays salary from the player resources for the unit
     * @post Exception guarantee: ?????????????
     */
    void paySalary();


    /**
     * @brief Checks if the unit is conquering or not.
     * @return Boolean value. True if the unit is conquering and false if not.
     * @post Exception guarantee: No-throw
     */
    bool isConqueringUnit();


    /**
     * @brief Sets the unit to be conquering or non-conquering
     * @param Boolean value. True if the unit is conquering and false if not.
     * @post Exception guarantee: No-throw
     */
    void setAsConquering(bool isConquering);


    /**
     * @brief Found in all classes inherited from UnitBase. All of them
     *        have their own description of their functionality
     */
    virtual ResourceMap getSalary() = 0;


    /**
     * @brief Found in all classes inherited from UnitBase. All of them
     *        have their own description of their functionality
     */
    virtual ResourceMap getCost() = 0;

private:

    std::weak_ptr<Student::GameSettingsManager> gameSettingsManager_;
    std::shared_ptr<Course::Coordinate> tileRelativeCoordinate_; //Units tile
                                                             //related coordinates

    std::weak_ptr<Course::TileBase> parentTile_; //Tile the unit is on

    bool isConqueringUnit_; //Tells if the unit is conquering (attacking) or not

}; // class WorkerBase

} // namespace Course


#endif // WORKERBASE_H

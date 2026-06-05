/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: tilebase.h , header for TileBase-class                       #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef TILEBASE_H
#define TILEBASE_H

#include <QPixmap>

#include "Core/gameobject.h"
#include "Core/basicresources.h"
#include "Core/playerbase.h"
#include "Buildings/buildingbase.h"
#include "Buildings/headquarters.h"
#include "DAL/gamesettingsmanager.h"
#include "Interfaces/igameeventhandler.h"
#include "Interfaces/iobjectmanager.h"
#include "Interfaces/ipressableobject.h"
#include "Tiles/tilebase.h"
#include "Units/unitbase.h"
#include "Core/descriptionmaps.h"


namespace Course {

/**
 * @brief The TileBase class is a base-class for different Tile-objects
 * in the game.
 *
 * Buildings can be placed on the tile depending on the tile.
 * Six units can be placed on any tile (conquering plus non-conquering)
 */

class TileBase : public GameObject, public Student::iPressableObject
{
public:
    const unsigned int MAX_UNITS;
    const ResourceMap BASE_PRODUCTION;

    /**
     * @brief Disabled parameterless constructor.
     */
    TileBase() = delete;


    /**
     * @brief Constructor for the class.
     *
     * @param location is the Coordinate where the Tile is located in the game.
     * @param size_x is how many maps grids wide the tile is (1)
     * @param size_y is how many maps grids tall the tile is (1)
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param max_unit tells how many units the tile can have as
     *        conquering or non-conquering.
     * @param production is the production of the tile (nothing)
     * @param basic_description is the description of the tile
     */
    TileBase(const Coordinate& location,
             int size_x,
             int size_y,
             const std::weak_ptr<iGameEventHandler>& eventhandler,
             const std::weak_ptr<iObjectManager>& objectmanager,
             const unsigned int& max_units,
             const ResourceMap& production,
             const std::string basic_description
             );


    /**
     * @brief Constructor for the class.
     *
     * @param location is the Coordinate where the Tile is located in the game.
     * @param size_x is how many maps grids wide the tile is (1)
     * @param size_y is how many maps grids tall the tile is (1)
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param max_unit tells how many units the tile can have as
     *        conquering or non-conquering.
     * @param production is the production of the tile (nothing)
     */
    TileBase(const Coordinate& location,
             int size_x,
             int size_y,
             const std::weak_ptr<iGameEventHandler>& eventhandler,
             const std::weak_ptr<iObjectManager>& objectmanager,
             const unsigned int& max_units,
             const ResourceMap& production
             );


    /**
     * @brief Default destructor.
     */
    virtual ~TileBase() = default;


    /**
     * @brief Returns the tile's type in string. In this case it's "TileBase"
     * @return Tile's type in string. In this case it's "TileBase"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief Adds an unit into the TileBase object. If the unit is in the
     *        player's own area it is added into units_ vector. Otherwise it
     *        is added into conqueringUnits_
     * @param Shared pointer of the unit to be added
     * @post Exception guarantee: ??????????????
     */
    virtual void addUnit(const std::shared_ptr<UnitBase>& unit);


    /**
     * @brief Removes an unit from the TileBase object and from the corresponding
     *        vector it is in (either units_ or conqueringUnits_)
     * @param Shared pointer of the unit to be removed
     * @post Exception guarantee: ??????????????
     */
    virtual void removeUnit(const std::shared_ptr<UnitBase>& unit);


    /**
     * @brief Adds a building on the TileBase object by storing it in
     *        the building_ shared pointer
     * @param Shared pointer of the building to be added
     * @post Exception guarantee: ??????????????
     */
    void addBuilding(const std::shared_ptr<BuildingBase>& building);


    /**
     * @brief Returns the shared pointer to the building the tile possibly has
     * @return Shared pointer of the building or nullptr
     * @post Exception guarantee: ??????????????
     */
    virtual std::shared_ptr<BuildingBase> getBuilding() const;
    

    /**
     * @brief Handles a conquer situation. If a tile has an unit but no one
     *        owns the tile the conquering player will get the tile. If
     *        the tile is owned by another player the tile will be conquered
     *        if the conquerer has more soldiers on the tile than the current
     *        owner (defender).
     * @param Shared pointer of the player who is conquering the tile
     * @post Exception guarantee: ??????????????
     */
    void conquerTile(std::shared_ptr<PlayerBase> currentPlayer);


    /**
     * @brief Checks if the tile has opponent's headquarters, The actual
     *        function is in Tiles/grassland.cpp
     * @param Shared pointer to the player that wants to know if the
     *        grassland has opponent's headquarters
     * @return Bool value. True if the tile has opponent's headquarters
     *         and false if not.
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual bool hasOpponentHeadquarters(std::shared_ptr<PlayerBase> player);


    /**
     * @brief Returns the increase of how many basic workers or experts the
     *        player can have according to the building the tile has.
     * @return Integer of the number of basic workers or experts the
     *         player can have.
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual int getMaxUnitsIncrease();


    /**
     * @brief Returns the increase of how many soldiers the player can have
     *        according to the building the tile has.
     * @return Integer of the number of soldiers the player can have.
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual int getMaxSoldiersIncrease();


    /**
     * @brief Returns the number of normal units on the tile
     * @return Unsigned integer of the number of normal units on the tile
     * @post Exception guarantee: No-throw
     */
    virtual unsigned int getUnitCount() const;


    /**
     * @brief Returns the number of conquering units on the tile
     * @return Unsigned integer of the number of conquering units on the tile
     * @post Exception guarantee: No-throw
     */
    virtual unsigned int getConqueringUnitCount() const;


    /**
     * @brief Returns the number of own soldiers on the tile
     * @return Unsigned integer of the number of soldiers on the tile
     * @post Exception guarantee: No-throw
     */
    virtual unsigned int getSoldierCount() const;


    /**
     * @brief Returns the number of opponent's soldiers on the tile
     * @return Unsigned integer of the number of opponent's soldiers on the tile
     * @post Exception guarantee: No-throw
     */
    virtual unsigned int getOpponentSoldierCount() const;


    /**
     * @brief Returns a vector of shared pointers to the non-conquering units
     *        that are in the tile.
     * @return A vector of shared pointers to the non-conquering units
     *         that are in the tile.
     * @post Exception guarantee: No-throw
     */
    virtual std::vector<std::shared_ptr<UnitBase>> getUnits() const;


    /**
     * @brief Returns a vector of shared pointers to the conquering units
     *        that are in the tile.
     * @return A vector of shared pointers to the conquering units
     *         that are in the tile.
     * @post Exception guarantee: No-throw
     */
    virtual std::vector<std::shared_ptr<UnitBase>> getConqueringUnits() const;


    /**
     * @brief Updates the animation of an object that might be
     *        in Grassland or River tile.
     * @post Exception guarantee: No-throw
     */
    virtual void updateAnimation();


    /**
     * @brief Returns a resource map of the expenses the tile
     *        will reduce from the player when the turn ends.
     * @return Resource map of the expenses the tile will reduce from
     *         the player when the turn ends.
     * @post Exception guarantee: ?????????????????????????????
     */
    ResourceMap getCurrentExpenses();


    /**
     * @brief Returns a resource map of the net expenses/revenues the tile
     *        will add/reduce from the player when the turn ends.
     * @return Resource map of the net expenses/revenues the tile will
     *         add/reduce from the player when the turn ends.
     * @post Exception guarantee: ?????????????????????????????
     */
    ResourceMap getCurrentNet();


    /**
     * @brief Updates the tile relative coordinates of the units.
     *        The tile has a 3x2 grid where the units can be placed.
     * @post Exception guarantee: ???????????????????
     */
    virtual void updateUnitCoordinates();


    /**
     * @brief Returns the four neighbouring tiles of the tile
     * @return Vector of the shared pointers of the neighbouring tiles
     * @post Exception guarantee: ???????????????????
     */
    std::vector<std::shared_ptr<Course::TileBase>>
                                        getNeighbourFourTiles();

    /**
     * @brief Returns the all neighbouring tiles of the tile
     * @return Vector of the shared pointers of the neighbouring tiles
     * @post Exception guarantee: ???????????????????
     */
    std::vector<std::shared_ptr<TileBase> > getNeighbourTiles();


    /**
     * @brief Checks if the tile has space for non-conquering units
     * @return Boolean value. True if the tile has space for non-conquering units
     *         and false if not.
     * @post Exception guarantee: No-throw
     */
    virtual bool hasSpaceForUnits() const final;


    /**
     * @brief Checks if the tile has space for conquering units
     * @return Boolean value. True if the tile has space for conquering units
     *         and false if not.
     * @post Exception guarantee: No-throw
     */
    virtual bool hasSpaceForConqueringUnits() const final;


    /**
     * @brief Sets gameSettingsManager_ to the object
     * @param Shared pointer to the GameSettingsManager
     * @post Exception guarantee: No-throw
     */
    void setGameSettings
          (const std::shared_ptr<Student::GameSettingsManager> manager);


    /**
     * @brief Is called when the tile is clicked and calls eventHandler_ to
     *        continue the action there
     * @post Exception guarantee: ????????????????????????
     */
    virtual void clickAction() override;


    /**
     * @brief Finds out if the tile is neighbouring a tile that has a different
     *        owner or no owner at all. The function then forms a vector that has
     *        pixmaps of the colored borders that need to be
     *        drawn on top of the tile.
     * @return Vector of the pixmaps to be drawn on top of the tile
     * @post Exception guarantee: ???????????????????????????
     */
    virtual std::vector<QPixmap> getOwnerBorderPixmap();


    /**
     * @brief Sets basicDescription_ for the tile object
     * @param String of the description that want to be set
     * @post Exception guarantee: No-throw
     */
    virtual void addBasicDescription(std::string desc);


    /**
     * @brief Gets basicDescription_ for the tile object
     * @return Description as a string
     * @post Exception guarantee: No-throw
     */
    std::string getBasicDescription();


    /**
     * @brief This function is used by the menu to get information about the
     *        net production the tile has. The information is showed to the
     *        player as a text.
     * @return String of the net production the tile might have
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual std::string getNetDescription();


    /**
     * @brief Found in all classes inherited from TileBase. All of them
     *        have their own description of their functionality
     */
    virtual ResourceMap getCurrentRevenue() = 0;


    /**
     * @brief Found in all classes inherited from TileBase. All of them
     *        have their own description of their functionality
     */
    virtual std::string getExtraDescription() = 0;


    /**
     * @brief Found in all classes inherited from TileBase. All of them
     *        have their own description of their functionality
     */
    virtual std::vector<std::string> getBuildableBuildings() = 0;


    /**
     * @brief Found in all classes inherited from TileBase. All of them
     *        have their own description of their functionality
     */
    virtual void generateResources() = 0;



private:
    std::string basicDescription_;

    std::shared_ptr<BuildingBase> building_;
    std::weak_ptr<Student::GameSettingsManager> gameSettingsManager_;

protected:
    std::vector<std::shared_ptr<UnitBase>> units_;
    std::vector<std::shared_ptr<UnitBase>> conqueringUnits_;

}; // class TileBase

} // namespace Course


#endif // TILEBASE_H

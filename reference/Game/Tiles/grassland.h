/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: grassland.h, header for Grassland-class                      #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef GRASSLAND_H
#define GRASSLAND_H

#include "tilebase.h"

#include "Buildings/farm.h"
#include "Buildings/headquarters.h"
#include "Buildings/mikontalo.h"
#include "Buildings/outpost.h"
#include "Buildings/nuclearplant.h"
#include "Buildings/village.h"
#include "Graphics/sceneitem.h"


namespace Course {

/**
 * @brief The Grassland class represents a grassland tile in the gameworld.
 *
 * A farm, headquarters, nuclear power plant, outpost and village can
 * be built on the tile.
 */

class Grassland : public TileBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    Grassland() = delete;

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
    Grassland(const Coordinate& location,
              int size_x,
              int size_y,
              const std::weak_ptr<iGameEventHandler>& eventhandler,
              const std::weak_ptr<iObjectManager>& objectmanager,
              const unsigned int& max_work = 3,
              const ResourceMap& production = ConstResourceMaps::EMPTY);


    /**
     * @brief Default destructor.
     */
    virtual ~Grassland() = default;


    /**
     * @brief Returns the tile's type in string. In this case it's "Grassland"
     * @return Tile's type in string. In this case it's "Grassland"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief Returns the building types that can be built on the tile.
     * @return List of building types (as a string) in a vector.
     * @post Exception guarantee: No-throw
     */
    virtual std::vector<std::string> getBuildableBuildings() override;


    /**
     * @brief Gives resources to tiles owner. The generated resources depend
     *        on the building and unit(s) the tile has
     * @post Exception guarantee: ????????????????????????
     */
    virtual void generateResources() override;


    /**
     * @brief Returns a resource map of the revenue the grassland tile
     *        will produce when the turn ends.
     * @return Resource map of the revenue the grassland tile will produce
     *         when the turn ends
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual Course::ResourceMap getCurrentRevenue() override;


    /**
     * @brief This function is used by the menu to get extra information about the
     *        grassland tile. The information is showed to the player as a text.
     * @return String of the extra description the tile might have
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual std::string getExtraDescription() override;


    /**
     * @brief This function is used by the menu to get information about the
     *        net production the tile has. The information is showed to the
     *        player as a text.
     * @return String of the net production the tile might have
     * @post Exception guarantee: ?????????????????????????????
     */
    std::string getNetDescription() override;


    /**
     * @brief Checks if the tile has opponent's headquarters
     * @param Shared pointer to the player that wants to know if the
     *        grassland has opponent's headquarters
     * @return Bool value. True if the tile has opponent's headquarters
     *         and false if not.
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual bool hasOpponentHeadquarters
                                (std::shared_ptr<PlayerBase> player) override;


    /**
     * @brief Returns the increase of how many basic workers or experts the
     *        player can have according to the building the tile has
     * @return Integer of the number of basic workers or experts the
     *         player can have.
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual int getMaxUnitsIncrease() override;


    /**
     * @brief Returns the increase of how many soldiers the player can have
     *        according to the building the tile has
     * @return Integer of the number of soldiers the player can have.
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual int getMaxSoldiersIncrease() override;


    /**
     * @brief Updates nuclear power plant animation. The animation is on
     *        only when the power plant is operating
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual void updateAnimation() override;


}; // class Grassland

} // namespace Course


#endif // GRASSLAND_H

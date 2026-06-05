/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: forest.h, header for Forest-class                            #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef FOREST_H
#define FOREST_H

#include "tilebase.h"
#include "Core/resourcemaps.h"


namespace Course {

/**
 * @brief The Forest class represents a forest tile in the gameworld.
 *
 * Forest is the only way to get wood. Check Core/resourcemaps.h
 * FOREST_PRODUCTION and FOREST_CAPACITY for specific ResourceMap values.
 * If the forest is cut down, the same buildings can be built on the tile
 * as on the grassland. If nothing is built the forest will grow back
 */

class Forest : public TileBase
{
public:

    /**
     * @brief Disabled parameterless constructor.
     */
    Forest() = delete;


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
    Forest(const Coordinate& location,
           int size_x,
           int size_y,
           const std::weak_ptr<iGameEventHandler>& eventhandler,
           const std::weak_ptr<iObjectManager>& objectmanager,
           const unsigned int& max_unit = 3,
           const ResourceMap& production = ConstResourceMaps::EMPTY);


    /**
     * @brief Default destructor.
     */
    virtual ~Forest() = default;


    /**
     * @brief Returns the tile's type in string. In this case it's "Forest"
     * @return Tile's type in string. In this case it's "Forest"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief Returns the building types that can be built on the tile.
     * @return List of building types (as a string) in a vector.
     *         If the forest hasn't been cut down, the function
     *         returns an empty vector.
     * @post Exception guarantee: No-throw
     */
    virtual std::vector<std::string> getBuildableBuildings() override;


    /**
     * @brief Gives resources to the tiles owner. In this case the only
     *        resource generated is wood. Production is affected by the
     *        number of basic workers the forest tile has.
     * @post Exception guarantee: ????????????????????????
     */
    virtual void generateResources() override;


    /**
     * @brief Returns a resource map of the revenue the forest tile will produce
     *        when the turn ends.
     * @return Resource map of the revenue the forest tile will produce
     *         when the turn ends
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual Course::ResourceMap getCurrentRevenue() override;


    /**
     * @brief This function is used by the menu to get information about the
     *        forest tile. The information is showed to the player as a text.
     *        This information tells us how much wood the tile has
     *        left and how many rounds the tile has been empty (cut).
     * @return String of the extra description the tile might have
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual std::string getExtraDescription() override;

private:
    int woodLeft_; //Wood resources the forest has left before it's replenished
    int roundsStumpsHaveBeen_;


}; // class Forest

} // namespace Course


#endif // FOREST_H


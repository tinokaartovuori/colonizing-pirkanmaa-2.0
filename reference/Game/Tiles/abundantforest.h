/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: AbundantForest.h, header for AbundantForest-class            #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef ABUNDANTFOREST_H
#define ABUNDANTFOREST_H

#include "tilebase.h"
#include "Core/resourcemaps.h"


namespace Student {

/**
 * @brief The AbundantForest class represents a Abundant Forest tile in the gameworld.
 *
 * AbundantForest is the free way to get some money. Check Core/resourcemaps.h
 * ABUNDANT_FOREST_PRODUCTION for specific ResourceMap values.
 * Abundant Forest can be harvested every round for little money.
 */

class AbundantForest : public Course::TileBase
{
public:

    /**
     * @brief Disabled parameterless constructor.
     */
    AbundantForest() = delete;


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
    AbundantForest(const Course::Coordinate& location,
           int size_x,
           int size_y,
           const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
           const std::weak_ptr<Course::iObjectManager> &objectmanager,
           const unsigned int& max_unit = 3,
           const Course::ResourceMap& production = Course::ConstResourceMaps::EMPTY);


    /**
     * @brief Default destructor.
     */
    virtual ~AbundantForest() = default;


    /**
     * @brief Returns the tile's type in string. In this case it's "AbundantForest"
     * @return Tile's type in string. In this case it's "AbundantForest"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief Returns the building types that can be built on the tile.
     * @return List of building types (as a string) in a vector.
     *         If the AbundantForest hasn't been cut down, the function
     *         returns an empty vector.
     * @post Exception guarantee: No-throw
     */
    virtual std::vector<std::string> getBuildableBuildings() override;


    /**
     * @brief Gives resources to the tiles owner. In this case the only
     *        resource generated is wood. Production is affected by the
     *        number of basic workers the AbundantForest tile has.
     * @post Exception guarantee: ????????????????????????
     */
    virtual void generateResources() override;


    /**
     * @brief Returns a resource map of the revenue the AbundantForest tile will produce
     *        when the turn ends.
     * @return Resource map of the revenue the AbundantForest tile will produce
     *         when the turn ends
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual Course::ResourceMap getCurrentRevenue() override;


    /**
     * @brief This function is used by the menu to get information about the
     *        AbundantForest tile. The information is showed to the player as a text.
     *        This information tells us how much wood the tile has
     *        left and how many rounds the tile has been empty (cut).
     * @return String of the extra description the tile might have
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual std::string getExtraDescription() override;



}; // class AbundantForest

} // namespace Course


#endif // ABUNDANTFOREST_H


/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: mountain.h, header for Mountain-class                        #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef MOUNTAIN_H
#define MOUNTAIN_H

#include "tilebase.h"


namespace Student {

/**
 * @brief The Mountain class represents mountain in the gameworld.
 *     
 * A mine can be built on the tile.
 */

class Mountain : public Course::TileBase
{
public:
    static const unsigned int MAX_BUILDINGS;
    static const unsigned int MAX_UNITS;
    static const Course::ResourceMap BASE_PRODUCTION;
    /**
     * @brief Disabled parameterless constructor.
     */
    Mountain() = delete;

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
    Mountain(const Course::Coordinate& location,
             int size_x,
             int size_y,
             const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
             const std::weak_ptr<Course::iObjectManager> &objectmanager,
             const unsigned int& max_units = 3,
             const Course::ResourceMap& production =
             Course::ConstResourceMaps::EMPTY);


    /**
     * @brief Default destructor.
     */
    virtual ~Mountain() = default;


    /**
     * @brief Returns the tile's type in string. In this case it's "Mountain"
     * @return Tile's type in string. In this case it's "Mountain"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;
    

    /**
     * @brief Returns the building types that can be built on the tile. In this
     *        case the only building is "Mine"
     * @return List of building types (as a string) in a vector.
     * @post Exception guarantee: No-throw
     */
    virtual std::vector<std::string> getBuildableBuildings() override;


    /**
     * @brief Gives resources to the tiles owner. Production is affected by the
     *        number of basic workers and experts the tile has.
     * @post Exception guarantee: ????????????????????????
     */
    virtual void generateResources() override;


    /**
     * @brief Returns a resource map of the revenue the mountain tile will
     *        produce when the turn ends.
     * @return Resource map of the revenue the mountain tile will produce
     *         when the turn ends
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual Course::ResourceMap getCurrentRevenue() override;


    /**
     * @brief This function is used by the menu to get extra information about the
     *        mountain tile. The information is showed to the player as a text.
     * @return String of the extra description the tile might have
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual std::string getExtraDescription() override;

}; // class Mountain

} // namespace Student


#endif // MOUNTAIN_H

/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: river.h, header for River-class                              #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/



#ifndef RIVER_H
#define RIVER_H

#include "tilebase.h"


namespace Student {

/**
 * @brief The River class represents River in the gameworld.
 *
 * A bridge and a hydroelectric power plant can be built on the tile.
 * If the river is a "corner piece", nothing can be built on it.
 */

class River : public Course::TileBase
{
public:
    static const unsigned int MAX_BUILDINGS;
    static const unsigned int MAX_UNITS;
    static const Course::ResourceMap BASE_PRODUCTION;
    /**
     * @brief Disabled parameterless constructor.
     */
    River() = delete;

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
    River(const Course::Coordinate& location,
              int size_x,
              int size_y,
              const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
              const std::weak_ptr<Course::iObjectManager>& objectmanager,
              const unsigned int& max_units = 3,
              const Course::ResourceMap& production =
              Course::ConstResourceMaps::EMPTY);
    /**
     * @brief Default destructor.
     */
    virtual ~River() = default;


    /**
     * @brief Returns the tile's type in string. In this case it's "River"
     * @return Tile's type in string. In this case it's "River"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override; 


    /**
     * @brief Returns the building types that can be built on the tile. In this
     *        case the only buildings are "Bridge" and
     *        "Hydroelectric Power Plant". If the tile is a river corner piece
     *        there are no buildings that can be built.
     * @return List of building types (as a string) in a vector.
     * @post Exception guarantee: No-throw
     */
    virtual std::vector<std::string> getBuildableBuildings() override;


    /**
     * @brief Gives or removes resources to the tiles owner. This depends
     *        on the building that is on the tile
     * @post Exception guarantee: ????????????????????????
     */
    virtual void generateResources() override;


    /**
     * @brief Returns a resource map of the revenue the river tile will
     *        produce when the turn ends.
     * @return Resource map of the revenue the river tile will produce
     *         when the turn ends
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual Course::ResourceMap getCurrentRevenue() override;


    /**
     * @brief This function is used by the menu to get extra information about the
     *        river tile. The information is showed to the player as a text.
     * @return String of the extra description the tile might have
     * @post Exception guarantee: No-throw
     */
    virtual std::string getExtraDescription() override;


    /**
     * @brief Adds an unit on the tile
     * @param Shared pointer to the unit to be added on the tile
     * @post Exception guarantee: ???????????????????
     */
    virtual void addUnit(const std::shared_ptr<Course::UnitBase> &unit) override;


    /**
     * @brief Updates the tile relative coordinates of the units.
     *        The tile has a 3x2 grid where the units can be placed.
     *        The function also checks if the unit is in the water.
     *        If so, a unit with a life belt is drawn
     * @post Exception guarantee: ???????????????????
     */
    virtual void updateUnitCoordinates() override;


    /**
     * @brief Updates hydroelectric power plant animation. The animation is on
     *        only when the power plant is operating
     * @post Exception guarantee: ?????????????????????????????
     */
    virtual void updateAnimation() override;


    /**
     * @brief Returns the river orientation. One means the direction is
     *        north-south, two west-east and three that it is a curve
     * @return Integer of the river orientation
     * @post Exception guarantee: No-throw
     */
    int getRiverOrientation();


    /**
     * @brief Sets the river orientation. One means the direction is
     *        north-south, two west-east and three that it is a curve
     * @param Integer of the river orientation
     * @post Exception guarantee: ?????????????????????????????
     */
    void setRiverOrientation(int ori);


    /**
     * @brief Returns the river shape. The shapes are in compass directions
     *        for example NS (north-south) or NW (north-west)
     * @return String ove the river shape.
     * @post Exception guarantee: No-throw
     */
    std::string getRiverShape();


    /**
     * @brief Sets the river shape. The shapes are in compass directions
     *        for example NS (north-south) or NW (north-west)
     * @param String ove the river shape.
     * @post Exception guarantee: No-throw
     */
    void setRiverShape(std::string shape);


private:
    int riverOrientation_; //One means the direction is north-south,
                           //two west-east and three is a curve

    std::string riverShape_; //Shape of the river. The directions are
                             //in compass directions for example
                             //NS (north-south) or NW (north-west)

}; // class River

} // namespace Student


#endif // RIVER_H

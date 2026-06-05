/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: buildingbase.h, header for BuildingBase-class                #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef BUILDINGBASE_H
#define BUILDINGBASE_H

#include "Core/placeablegameobject.h"
#include "Core/basicresources.h"
#include "Core/descriptionmaps.h"

namespace Course {

class TileBase;

/**
 * @brief The BuildingBase class is a base-class for
 * different buildings in the game.
 *
 * * Can increase base-production for a Tile.
 * * Can call functions from GameEventHandler.
 *
 * Buildings can have hold-markers that prevent normal operation.
 */

class BuildingBase : public PlaceableGameObject
{
public:

    const ResourceMap BUILD_COST;
    const ResourceMap PRODUCTION_EFFECT;

    /**
     * @brief Disabled parameterless constructor.
     */
    BuildingBase() = delete;


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param owner points to the owning player.
     * @param buildcost is a ResourceMap of the building cost of the building
     * @param production is a ResourceMap of the production of the building
     * @param basic_description is a string of the buildings description
     *
     * @post Exception Guarantee: No guarantee.
     * @exception OwnerConflict - if the building conflicts with tile's
     * ownership.
     */
    explicit BuildingBase(const std::weak_ptr<iGameEventHandler>& eventhandler,
            const std::weak_ptr<iObjectManager>& objectmanager,
            const std::weak_ptr<PlayerBase>& owner,
            const ResourceMap& buildcost = {},
            const ResourceMap& production = {},
            const std::string& basic_description = ""
            );


    /**
     * @brief Default destructor.
     */
    virtual ~BuildingBase() = default;


    /**
     * @brief Returns the building's type in string.
     *        In this case it's "BuildingBase"
     * @return Building's type in string. In this case it's "BuildingBase"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief Returns a resource map of the buildings production
     * @return Resource map of the buildings production
     * @post Exception guarantee: No-throw
     */
    ResourceMap getProduction();


    /**
     * @brief Returns a resource map of the buildings building cost
     * @return Resource map of the buildings building cost
     * @post Exception guarantee: No-throw
     */
    virtual ResourceMap getCost() override;


    /**
     * @brief Sets description for the building object
     * @param String of the description to be set
     * @post Exception guarantee: No-throw
     */
    void addBasicDescription(std::string desc);


    /**
     * @brief Gets description of the building object
     * @return String of the description to be set
     * @post Exception guarantee: No-throw
     */
    std::string getBasicDescription();


    /**
     * @brief Sets parent tile for the building. This is called every time
     *        a building is added on a tile
     * @param Shared pointer to the parent tile
     * @post Exception guarantee: No-throw
     */
    void setParentTile(std::shared_ptr<TileBase> parentTile);


protected:
    std::string basicDescription_;
    std::weak_ptr<TileBase> parentTile_;


}; // class BuildingBase

} // namespace Course


#endif // BUILDINGBASE_H

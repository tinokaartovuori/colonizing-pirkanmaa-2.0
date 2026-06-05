/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: headquarters.h, header for HeadQuarters-class                #                                               #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef HEADQUARTERS_H
#define HEADQUARTERS_H

#include "buildingbase.h"
#include "Core/resourcemaps.h"


namespace Course {

/**
 * @brief The HeadQuarters class represents player's headquarters in the game.
 *
 * The headquarters is player's main building. If it gets
 * conquered the player loses.
 */

class HeadQuarters : public BuildingBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    HeadQuarters() = delete;


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param owner points to the owning player.
     *
     * @post Exception Guarantee: No guarantee.
     * @exception OwnerConflict - if the building conflicts with tile's
     * ownership.
     */
    explicit HeadQuarters(const std::weak_ptr<iGameEventHandler>& eventhandler,
            const std::weak_ptr<iObjectManager>& objectmanager,
            const std::weak_ptr<PlayerBase>& owner);


    /**
     * @brief Default destructor.
     */
    virtual ~HeadQuarters() = default;


    /**
     * @brief Returns the building's type in string.
     *        In this case it's "Headquarters"
     * @return Building's type in string. In this case it's "Headquarters"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief This function is used by the menu to get information about the
     *        headquarters building. The information is showed to the player
     *        as a text. This information tells us how much wood the tile has
     *        left and how many rounds the tile has been empty (cut).
     * @return String of the extra description the tile might have
     * @post Exception guarantee: ?????????????????????????????
     */
    std::string getExtraDescription();


    /**
     * @brief Marks the headquarters as conquered. The function is called when
     *        someone conquers the tile.
     * @post Exception guarantee: ?????????????????????????????
     */
    void setConquered();


    /**
     * @brief Checks if the headquarters has been conquered or not.
     * @return Boolean value. True if the headquarters
     *         is conquered and false if not
     * @post Exception guarantee: ?????????????????????????????
     */
    bool isConquered();

private:
    bool conqured_; //Tells if the HQ has been conquered or not


}; // class HeadQuarters

} // namespace Course


#endif // HEADQUARTERS_H

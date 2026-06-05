/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: playerbase.h, header for PlayerBase-class                    #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef PLAYERBASE_H
#define PLAYERBASE_H

#include <string>
#include <vector>
#include <memory>

#include "gameobject.h"
#include "resourcemaps.h"
#include "Tiles/tilebase.h"
#include "Units/unitbase.h"

#include "Graphics/gamescene.h"
#include "Interfaces/iobjectmanager.h"

namespace Course {
/**
 * @class PlayerBase
 * @brief The PlayerBase class is a base class for classes used to describe
 * a player in game.
 *
 * The class can be used to store and access GameObjects.
 * Expired weak pointers are automatically removed when requesting or removing
 * objects.
 *
 * @note Objects are stored as weak pointers.
 *
 *
 */

//#ifndef COURSE_OBJECTID
//#define COURSE_OBJECTID
using ObjectId = unsigned int;
//#endif


class PlayerBase
{
public:
    /**
     * @brief Constructor for the class
     * @param name A std::string for player's name
     * @param objects (optional) A std::vector of shared-pointers to
     * GameObjects.
     */
    PlayerBase(const std::string& name,
               int playerNum,
               std::weak_ptr<Course::iObjectManager> objectmanager);

    /**
     * @brief Default destructor
     */
    ~PlayerBase() = default;



    /**
     * @brief Stores a weak GameObject-pointer.
     * @param object Is a weak pointer to the stored GameObject
     * @post Exception guarantee: Strong
     * @exception See std::vector::push_back()
     */
    void addObject(std::shared_ptr<GameObject> object);

    bool hasObject(std::shared_ptr<GameObject> object);

    /**
     * @brief Stores a vector of weak GameObject-pointers.
     * @param objects Is an std::vector of weak GameObject-pointers.
     * @post Exception guarantee: Strong
     * @exception See std::vector::insert()
     */
    void addObjects(const std::vector<std::shared_ptr<GameObject>> &objects);

    void addOrRemoveResources(ResourceMap resources);

    ResourceMap getResources();

    bool hasEnoughResources(ResourceMap cost);


    /**
     * @brief Removes a weak GameObject-pointer and expired weak pointers
     * @param object a weak pointer to GameObject
     * @post Exception guarantee: Basic
     * @exception ExpiredPointer - object is expired
     * @exception KeyError - No objects match the searched object
     */
    void removeObject(const std::shared_ptr<GameObject>& object);

    /**
     * @brief Removes a list of weak GameObject-pointers and
     * expired weak pointers.
     * @param objects A vector of weak GameObject-pointers
     * @post Exception guarantee: No-throw
     * @note If some of the provided weak pointers are expired,
     * no exceptions are thrown.
     */
    void removeObjects(
            const std::vector<std::shared_ptr<GameObject> >& objects);

    /**
     * @brief Removes a weak GameObject-pointers based on a ObjectId
     * and removes expired weak pointers.
     * @param id An ObjectId (unsigned int) for GameObject which is removed
     * @post Exception guarantee: Basic
     * @exception KeyError - No GameObjects with given ID were found.
     * @exception See std::remove_if
     */
    void removeObject(const ObjectId& id);

    /**
     * @brief Removes a list of weak GameObject-pointers based on a ObjectId
     * and removes expired weak pointers.
     * @param objects A vector of ObjectId (unsigned int) for GameObjects
     *  that are removed
     * @post Exception guarantee: No-throw
     * @note Even if some of the provided ID's are not found,
     * no exceptions are thrown.
     */
    void removeObjects(const std::vector<ObjectId>& objects);

    // Getters
    /**
     * @brief Returns the vector of weak GameObject-pointers
     *  that are currently stored in the Player-object.
     * @return Copy of m_objects -vector
     * @post Exception guarantee: Strong
     */
    std::vector<std::shared_ptr<GameObject> > getObjects() const;

    /**
     * @brief Returns the Player-object's name
     * @return Copy of current string in m_name
     * @post Exception guarantee: No-throw
     */
    std::string getName() const;

    /**
     * @brief Returns the Player-object's player number
     * @return Copy of current string in playerNum_
     * @post Exception guarantee: No-throw
     */
    int getPlayerNum() const;

    int getFreeUnitAmount();

    int getFreeSoldierAmount();

    int getMaxUnitAmount();

    int getMaxSoldierAmount();

    void updateUnitAmounts();

    void eliminateExcessUnits();

    int getCurrentUnitAmount();

    int getCurrentBasicWorkerAmount();

    int getCurrentExpertAmount();

    int getCurrentSoldierAmount();

    void limitResources();


private:
    std::string m_name;
    int playerNum_;
    std::weak_ptr<Course::iObjectManager> objectManager_;
    std::vector<std::weak_ptr<GameObject> > objects_;
    int maxSoldierAmount_;
    ResourceMap resources_;

    int maxUnitAmount_;

};

}
#endif // PLAYERBASE_H

/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: gameobject.h, header for GameObect-class                     #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef GAMEOBJECT_H
#define GAMEOBJECT_H

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "coordinate.h"
#include "baseobject.h"

namespace Course {

// Forward declarations

// Some aliases to make things easier
#ifndef COURSE_OBJECTID
#define COURSE_OBJECTID
using ObjectId = unsigned int;
#endif
/**
 *@brief DescriptionMap is an alias for std::map<std::string, std::string>
 */
using DescriptionMap = std::map<std::string, std::string>;

/**
 * @brief The GameObject class is base-class that contain's general information
 * on different Objects in game.
 *
 * * ID
 * * Possible owner
 * * Possible location coordinate
 * * Metadata in a string->string map
 * * Pointers to GameEventHandler and ObjectManager
 *
 * @note The functions consist mainly of get and set -functions that are used
 * to access and store the information specified above.
 *
 */
class GameObject : public BaseObject,
        public std::enable_shared_from_this<GameObject>
{
public:

    GameObject(const Coordinate& coordinate,
               int width,
               int height,
               const std::weak_ptr<iGameEventHandler>& eventhandler,
               const std::weak_ptr<iObjectManager>& objectmanager);


    /**
     * @brief GameObject constructor that can specify an owner.
     * @param owner a shared pointer to player that "owns" the object.
     * @param eventhandler a shared pointer to Game's GameEventHandler.
     * @param objectmanager a shared pointer to Game's ObjectManager.
     */
    explicit GameObject(const std::weak_ptr<PlayerBase>& owner,
               const std::weak_ptr<iGameEventHandler>& eventhandler,
               const std::weak_ptr<iObjectManager>& objectmanager);

    /**
     * @brief GameObject constructor that can specify a coordinate and an owner.
     * @param coordinate a shared pointer to coordinates where the object is
     * located.
     * @param owner a shared pointer to player that "owns" the object.
     * @param eventhandler a shared pointer to Game's GameEventHandler.
     * @param objectmanager a shared pointer to Game's ObjectManager.
     */
    explicit GameObject(const Coordinate& coordinate,
               const std::weak_ptr<PlayerBase>& owner,
               const std::weak_ptr<iGameEventHandler>& eventhandler,
               const std::weak_ptr<iObjectManager>& objectmanager);

    /**
     * @brief GameObject constructor that can specify a coordinate and an owner.
     * @param coordinate a shared pointer to coordinates where the object is
     * located.
     * @param owner a shared pointer to player that "owns" the object.
     * @param eventhandler a shared pointer to Game's GameEventHandler.
     * @param objectmanager a shared pointer to Game's ObjectManager.
     */
    explicit GameObject(const Coordinate& coordinate,
               int width,
               int height,
               const std::weak_ptr<PlayerBase>& owner,
               const std::weak_ptr<iGameEventHandler>& eventhandler,
               const std::weak_ptr<iObjectManager>& objectmanager);

    /**
     * @brief ~GameObject Default destructor.
     */
    virtual ~GameObject() = default;

    /**
     * @brief Change GameObject's "owner".
     * @param owner a shared pointer to the new "owner".
     * @post Exception guarantee: No-throw
     */
    void setOwner(const std::shared_ptr<PlayerBase> &owner);


    /**
     * @brief Returns GameObject's owner.
     * @return A shared-pointer to GameObject's owner
     * @post Exception guarantee: No-throw
     */
    std::shared_ptr<PlayerBase> getOwner() const;


    /**
     * @brief getType Returns a string describing objects type.
     * This should be overriden in each inherited class.
     * Makes checking object's type easier for students.
     * @return std::string that represents Object's type.
     * @post Exception guarantee: No-throw
     * @note You can use this in e.g. debugging and similar printing.
     */
    virtual std::string getType() const override;


    /**
     * @brief Function to compare if objects have same owner.
     * * @param other The other GameObject
     * @return True - If owners match or both are null
     * False - If owners don't match
     * @post Excepetion guarantee: Strong
     * @exception ExpiredPointer - If any owner weak_ptr has expired.
     */
    bool hasSameOwnerAs(
            const std::shared_ptr<GameObject>& other) const;


protected:

    std::weak_ptr<PlayerBase> owner_;

};

}
#endif // GAMEOBJECT_H

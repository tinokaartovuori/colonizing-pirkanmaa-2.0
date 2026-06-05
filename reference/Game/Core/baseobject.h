/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: baseobject.h, header for BaseObect-class                     #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef BASEOBJECT_H
#define BASEOBJECT_H

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "coordinate.h"
#include "Graphics/animationoption.h"
#include "Graphics/imagevectors.h"


namespace Student {
class GameSettingsManager;
}

namespace Course {

// Forward declarations
class PlayerBase;
class GameObject;
class iObjectManager;
class iGameEventHandler;
class SceneItem;

// Some aliases to make things easier
//#ifndef COURSE_OBJECTID
//#define COURSE_OBJECTID
using ObjectId = unsigned int;
//#endif

/**
 *@brief DescriptionMap is an alias for std::map<std::string, std::string>
 */
using DescriptionMap = std::map<std::string, std::string>;

/**
 * @brief The BaseObject class is base-class that contain's general information
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
class BaseObject
{
public:
    /**
     * @brief ID is a constant value that can be used to identify
     * BaseObjects through ID numbers.
     */
    const ObjectId ID;

    /**
     * @brief A simple copy-constructor for BaseObject
     * @param original is the original BaseObject
     */
    BaseObject(const BaseObject& original);

    /**
     * @brief Constructor that only specifies GameEventHandler and ObjectManager
     * @param eventhandler a shared pointer to Game's GameEventHandler.
     * @param objectmanager a shared pointer to Game's ObjectManager.
     */
    BaseObject(const std::weak_ptr<iGameEventHandler>& eventhandler,
               const std::weak_ptr<iObjectManager>& objectmanager);


    /**
     * @brief BaseObject constructor that can specify a coordinate.
     * @param coordinate a shared pointer to coordinates where the object is
     * located.
     * @param eventhandler a shared pointer to Game's GameEventHandler.
     * @param objectmanager a shared pointer to Game's ObjectManager.
     */
    BaseObject(const Coordinate& coordinate,
               const std::weak_ptr<iGameEventHandler>& eventhandler,
               const std::weak_ptr<iObjectManager>& objectmanager);

    /**
     * @brief BaseObject constructor that can specify a coordinate.
     * @param coordinate a shared pointer to coordinates where the object is
     * located.
     * @param eventhandler a shared pointer to Game's GameEventHandler.
     * @param objectmanager a shared pointer to Game's ObjectManager.
     */
    BaseObject(const Coordinate& coordinate,
               int width,
               int height,
               const std::weak_ptr<iGameEventHandler>& eventhandler,
               const std::weak_ptr<iObjectManager>& objectmanager);

    /**
     * @brief ~BaseObject Default destructor.
     */
    virtual ~BaseObject() = default;


    int getID();


    /**
     * @brief Change BaseObject's coordinate with a shared pointer to a
     * coordinate.
     * @param coordinate A shared-pointer to the new coordinate.
     * @post Exception guarantee: No-throw
     * @note This creates new Coordinate based on the coordinate.
     * The Coordinate can't be modified from outside of the class.
     * @note Can be used to unset coordinate with null-shared-pointer.
     */
    void setCoordinate(
            const std::shared_ptr<Coordinate>& coordinate);

    /**
     * @brief Change BaseObject's coordinate.
     * @param coordinate The new coordinate.
     * @post Exception guarantee: No-throw
     */
    void setCoordinate(const Coordinate& coordinate);

    /**
     * @brief Returns a pointer to a copy of BaseObject's coordinate.
     * @return Shared-pointer copy of the BaseObject's coordinate,
     * if the BaseObject has a coordinate.
     * Null-shared-pointer if the BaseObject has no coordinate.
     * @post Exception guarantee: Strong
     * @exception See std::make_shared
     * @note To change BaseObject's coordinate you must use setCoordinate.
     * This prevent unwanted alterations by accident.
     */
    std::shared_ptr<Coordinate> getCoordinatePtr() const;

    /**
     * @brief Returns BaseObject's current coordinate.
     * @post Exception guaranee: Strong
     * @exception
     * InvalidPointer - If the BaseObject doesn't have a coordinate.
     */
    Coordinate getCoordinate() const;

    /**
     * @brief getType Returns a string describing objects type.
     * @return std::string that represents Object's type.
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const;


    void setImageFiles(std::vector<std::string> imageVector);

    /**
     * @brief getImageFile Returns a string for the objects image file path.
     * @return std::string that represents Object's image file path
     * @post Exception guarantee: No-throw
     */
    std::vector<std::string> getImageFiles() const;

    void setAnimationOption(Student::AnimationOption option);

    Student::AnimationOption getAnimationOption();



    /**
     * @brief has_same_coordinate
     * @param other The other BaseObject
     * @return \n
     * True - If coordinates match or both are null \n
     * False - If the coordinates don't match \n
     * @post Exception guarantee: Strong
     */
    bool hasSameCoordinateAs(
            const std::shared_ptr<BaseObject>& other) const;


    int getWidth();

    int getHeight();

protected:

    /**
     * @brief This is the primary method for locking GameEventHandler inside
     * different BaseObject-classes.
     * @return shared pointer to the GameEventHandler
     * @post Exception guarantee: No-throw
     */
    virtual std::shared_ptr<iGameEventHandler> lockEventHandler() const final;
    /**
     * @brief This is the primary method for locking ObjectManager inside
     * different BaseObject classes.
     * @return shared pointer to the ObjectManager
     * @post Exception guarantee: No-throw
     */
    virtual std::shared_ptr<iObjectManager> lockObjectManager() const final;

private:

    const std::weak_ptr<iGameEventHandler> EVENTHANDLER;
    const std::weak_ptr<iObjectManager> OBJECTMANAGER;

    std::unique_ptr<Coordinate> coordinate_;

    std::vector<std::string> imageFilePaths_;
    Student::AnimationOption m_animation_option;

    static ObjectId c_next_id;

protected:
    int m_width;
    int m_height;

};

}
#endif // BASEOBJECT_H

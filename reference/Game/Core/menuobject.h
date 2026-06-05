/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: menuobject.h, header for MenuObject-class                    #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef MENUOBJECT_H
#define MENUOBJECT_H

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "coordinate.h"
#include "baseobject.h"

namespace Student {

// Forward declarations

// Some aliases to make things easier
#ifndef COURSE_OBJECTID
#define COURSE_OBJECTID
/**
 *@brief ObjectId is an alias for unsigned int
 */
using ObjectId = unsigned int;
#endif
/**
 *@brief DescriptionMap is an alias for std::map<std::string, std::string>
 */
using DescriptionMap = std::map<std::string, std::string>;

/**
 * @brief The MenuObject class is base-class that contain's general information
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
class MenuObject : public Course::BaseObject
{
public:

    MenuObject() = delete;

    MenuObject(const Course::Coordinate &coordinate,
          int width,
          int height,
          const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
          const std::weak_ptr<Course::iObjectManager> &objectmanager);


    MenuObject(const Course::Coordinate& coordinate,
          const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
          const std::weak_ptr<Course::iObjectManager>& objectmanager);

    /**
     * @brief ~MenuObject Default destructor.
     */
    virtual ~MenuObject() = default;


    virtual std::string getType() const override;

    void addToAbsoluteCoordinate(QPoint coord);

    QPoint getAbsoluteCoordinates();
    
    void multiPixMap(bool onoff);

    bool isMultiPixMap();

    void inverseMultiPixMap(bool onoff);

    bool isInverseMultiPixMap();

protected:
    QPoint absoluteCoordinate;
    bool isMultiPixMap_;
    bool isInverseMultiPixMap_;

};

}
#endif // MENUOBJECT_H

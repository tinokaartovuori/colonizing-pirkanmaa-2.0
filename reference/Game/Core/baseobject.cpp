/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: baseobject.cpp, see baseobject.h for more info               #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "baseobject.h"
#include "Interfaces/igameeventhandler.h"
#include "Interfaces/iobjectmanager.h"
#include "Exceptions/keyerror.h"
#include "Exceptions/invalidpointer.h"
#include "playerbase.h"

#include <QDebug>
#include <QGraphicsItem>

#include <algorithm>

namespace Course {

// Private static variables must be initialized this way.
unsigned int BaseObject::c_next_id = 0;

BaseObject::BaseObject(const BaseObject &original):
    ID(BaseObject::c_next_id),
    EVENTHANDLER(original.EVENTHANDLER),
    OBJECTMANAGER(original.OBJECTMANAGER)
{
    coordinate_ = std::make_unique<Coordinate>(*original.coordinate_);
    ++c_next_id;
    m_animation_option = Student::AnimationOption(AnimationOptions::EMPTY);
    m_width = 1;
    m_height = 1;
}

BaseObject::BaseObject(const std::weak_ptr<iGameEventHandler>& eventhandler,
                       const std::weak_ptr<iObjectManager>& objectmanager):
    ID(BaseObject::c_next_id),
    EVENTHANDLER(eventhandler),
    OBJECTMANAGER(objectmanager),
    coordinate_()
{
    ++c_next_id;
    m_animation_option = Student::AnimationOption();
    m_width = 1;
    m_height = 1;
}

BaseObject::BaseObject(const Coordinate& coordinate,
                       const std::weak_ptr<iGameEventHandler>& eventhandler,
                       const std::weak_ptr<iObjectManager>& objectmanager):
    ID(BaseObject::c_next_id),
    EVENTHANDLER(eventhandler),
    OBJECTMANAGER(objectmanager),
    coordinate_()
{
    coordinate_ = std::make_unique<Coordinate>(coordinate);
    ++c_next_id;
    m_animation_option = Student::AnimationOption();
    m_width = 1;
    m_height = 1;
}

BaseObject::BaseObject(const Coordinate& coordinate,
                       int width,
                       int height,
                       const std::weak_ptr<iGameEventHandler>& eventhandler,
                       const std::weak_ptr<iObjectManager>& objectmanager):
    ID(BaseObject::c_next_id),
    EVENTHANDLER(eventhandler),
    OBJECTMANAGER(objectmanager),
    coordinate_(),
    m_width(width),
    m_height(height)
{
    coordinate_ = std::make_unique<Coordinate>(coordinate);
    ++c_next_id;
    m_animation_option = Student::AnimationOption();
}

int BaseObject::getID()
{
    return ID;
}

void BaseObject::setCoordinate(const std::shared_ptr<Coordinate>& coordinate)
{
    coordinate_ = std::make_unique<Coordinate>(*coordinate);
}


void BaseObject::setCoordinate(const Coordinate& coordinate)
{
    coordinate_ = std::make_unique<Coordinate>(coordinate);
}


std::shared_ptr<Coordinate> BaseObject::getCoordinatePtr() const
{
    if(coordinate_)
    {
        return std::make_shared<Coordinate>(*coordinate_);
    }
    return nullptr;
}


Coordinate BaseObject::getCoordinate() const
{
    if( not coordinate_ )
    {
        throw InvalidPointer("BaseObject has no Coordinate.");
    }
    return Coordinate(*coordinate_);
}


std::string BaseObject::getType() const
{
    return "BaseObject";
}


void BaseObject::setImageFiles(std::vector<std::string> imageVector)
{
    imageFilePaths_ = imageVector;
}


std::vector<std::string> BaseObject::getImageFiles() const
{
    return imageFilePaths_;
}


void BaseObject::setAnimationOption(Student::AnimationOption option)
{
    m_animation_option = option;
}


Student::AnimationOption BaseObject::getAnimationOption()
{
    return m_animation_option;
}


bool BaseObject::hasSameCoordinateAs(
        const std::shared_ptr<BaseObject> &other) const
{
    if(not other)
    {
        return false;
    }
    if(not coordinate_)
    {
        if(not other->getCoordinatePtr())
        {
            return true;
        }
        return false;
    }

    return coordinate_->operator ==(*(other->getCoordinatePtr().get()));
}


int BaseObject::getWidth()
{
    return m_width;
}


int BaseObject::getHeight()
{
    return m_height;
}


std::shared_ptr<iGameEventHandler> BaseObject::lockEventHandler() const
{
    std::shared_ptr<iGameEventHandler> handler = EVENTHANDLER.lock();
    Q_ASSERT(handler);
    return handler;
}


std::shared_ptr<iObjectManager> BaseObject::lockObjectManager() const
{
    std::shared_ptr<iObjectManager> handler = OBJECTMANAGER.lock();
    Q_ASSERT(handler);
    return handler;
}


}


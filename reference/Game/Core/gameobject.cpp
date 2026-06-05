/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: gameobject.cpp, see gameobject.h for more info               #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "gameobject.h"
#include "Interfaces/igameeventhandler.h"
#include "Interfaces/iobjectmanager.h"
#include "Exceptions/keyerror.h"
#include "Exceptions/invalidpointer.h"
#include "playerbase.h"

#include <QDebug>


#include <algorithm>

namespace Course {


GameObject::GameObject(const Coordinate& coordinate,
                       int width,
                       int height,
                       const std::weak_ptr<iGameEventHandler>& eventhandler,
                       const std::weak_ptr<iObjectManager>& objectmanager):
    BaseObject(coordinate, width, height, eventhandler, objectmanager), owner_()
{
}

GameObject::GameObject(const std::weak_ptr<PlayerBase>& owner,
                       const std::weak_ptr<iGameEventHandler>& eventhandler,
                       const std::weak_ptr<iObjectManager>& objectmanager):
    BaseObject(eventhandler, objectmanager), owner_(owner)
{
}

GameObject::GameObject(const Coordinate& coordinate,
                       const std::weak_ptr<PlayerBase>& owner,
                       const std::weak_ptr<iGameEventHandler>& eventhandler,
                       const std::weak_ptr<iObjectManager>& objectmanager):
    BaseObject(coordinate, eventhandler, objectmanager), owner_(owner)
{
}

GameObject::GameObject(const Coordinate& coordinate,
                       int width,
                       int height,
                       const std::weak_ptr<PlayerBase>& owner,
                       const std::weak_ptr<iGameEventHandler>& eventhandler,
                       const std::weak_ptr<iObjectManager>& objectmanager):
    BaseObject(coordinate, width, height, eventhandler, objectmanager), owner_(owner)

{
}

void GameObject::setOwner(const std::shared_ptr<PlayerBase> &owner)
{

    //Object has a previous owner so it is removed from the previous owner
    if (owner_.lock() != nullptr && owner_.lock() != owner) {
        owner_.lock()->removeObject(shared_from_this());
    }

    //The new owner doesn't have the object so it is added to the owner
    if (owner != nullptr && !owner->hasObject(shared_from_this())) {
        owner->addObject(shared_from_this());
    }

    owner_ = std::weak_ptr<PlayerBase>(owner);
}


std::shared_ptr<PlayerBase> GameObject::getOwner() const
{
    return owner_.lock();
}


std::string GameObject::getType() const
{
    return "GameObject";
}

bool GameObject::hasSameOwnerAs(
        const std::shared_ptr<GameObject> &other) const
{
    return (getOwner().get() == other->getOwner().get());
}

}


/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: playerbase.cpp, see playerbase.h for more info               #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "playerbase.h"

#include <algorithm>
#include "Exceptions/keyerror.h"
#include <QDebug>


namespace Course{


PlayerBase::PlayerBase(const std::string& name,
                       int playerNum,
                       std::weak_ptr<Course::iObjectManager> objectmanager):
    m_name(name),
    playerNum_(playerNum),
    objectManager_(objectmanager),
    objects_(),
    maxSoldierAmount_(0),
    maxUnitAmount_(0)
{
    addOrRemoveResources(Course::ConstResourceMaps::STARTING_RESOURCES);
}


void PlayerBase::addObject(std::shared_ptr<GameObject> object)
{
    objects_.push_back(std::weak_ptr<GameObject>(object));
}


bool PlayerBase::hasObject(std::shared_ptr<GameObject> object)
{
    for (auto o : objects_) {
        if (o.lock() == object) {
            return true;
        }
    }
    return false;

}


void PlayerBase::addObjects(
        const std::vector<std::shared_ptr<GameObject> >& objects)
{
    objects_.insert(objects_.end(), objects.begin(), objects.end());
}


void PlayerBase::addOrRemoveResources(ResourceMap resources)
{
    resources_ = mergeResourceMaps(resources_, resources);

}


ResourceMap PlayerBase::getResources()
{
    return resources_;
}


bool PlayerBase::hasEnoughResources(ResourceMap cost)
{
    ResourceMap resources = mergeResourceMaps(resources_, cost);

    for (auto key : resources) {
        if (key.second < 0) {
            return false;
        }
    }

    return true;
}


void PlayerBase::removeObject(const std::shared_ptr<GameObject>& object)
{
    if( not object )
    {
        removeObject(std::numeric_limits<ObjectId>::max());
    }
    removeObject(object->ID);
}


void PlayerBase::removeObject(const ObjectId& id)
{
    bool found = false;
    // Use find if to do weak_ptr locking inside seach-function
    //  for ID-recognition.
    auto it = std::remove_if(objects_.begin(), objects_.end(),
                           [id, &found](std::weak_ptr<GameObject>& x){
            auto locked = x.lock();
            if( not locked )
            {
                return true;
            }
            if( locked->ID == id )
            {
                found = true;
                return true;
            }
            return false;
            });

    objects_.erase(it, objects_.end());

    if(not found)
    {
        throw KeyError("Object not found.");
    }
}


void PlayerBase::removeObjects(
        const std::vector<std::shared_ptr<GameObject> >& objects)
{
    for( auto it = objects.begin(); it != objects.end(); ++it)
    {
        try{
            removeObject(*it);
        }
        catch (const KeyError&){
            continue;
        }
    }
}


void PlayerBase::removeObjects(const std::vector<ObjectId>& objects)
{
    for( auto it = objects.begin(); it != objects.end(); ++it)
    {
        try{
            removeObject(*it);
        }
        catch(const KeyError&){
            continue;
        }
    }
}


std::vector<std::shared_ptr<GameObject> > PlayerBase::getObjects() const
{
    std::vector<std::shared_ptr<GameObject> > objects;
    for(auto it = objects_.begin(); it != objects_.end(); ++it)
    {
        if(not it->expired())
        {
            objects.push_back(it->lock());
        }
    }

    return objects;
}


std::string PlayerBase::getName() const
{
    return m_name;
}


int PlayerBase::getPlayerNum() const
{
    return playerNum_;
}

int PlayerBase::getFreeUnitAmount()
{

    return maxUnitAmount_ - getCurrentBasicWorkerAmount() - getCurrentExpertAmount();
}

int PlayerBase::getFreeSoldierAmount()
{
    return maxSoldierAmount_ - getCurrentSoldierAmount();
}

int PlayerBase::getMaxUnitAmount()
{
    updateUnitAmounts();

    return maxUnitAmount_;
}

int PlayerBase::getMaxSoldierAmount()
{
    updateUnitAmounts();

    return maxSoldierAmount_;
}

void PlayerBase::updateUnitAmounts()
{
    int newMaxUnitAmount = 0;
    int newMaxSoldierAmount = 0;
    for (auto o : objects_) {
        if (std::dynamic_pointer_cast<Course::TileBase>(o.lock()) != nullptr) {
            std::shared_ptr<Course::TileBase> tile =
                    std::dynamic_pointer_cast<Course::TileBase>(o.lock());
            newMaxUnitAmount += tile->getMaxUnitsIncrease();
            newMaxSoldierAmount += tile->getMaxSoldiersIncrease();
        }
    }


    // Limits the maxUnitAmount_ to Unit Limit from resource maps
    if (newMaxUnitAmount >= Course::ConstResourceMaps::UNIT_LIMITS) {
        newMaxUnitAmount = Course::ConstResourceMaps::UNIT_LIMITS;
    }

    // Limits the maxSoldierAmount_ to Unit Limit from resource maps
    if (newMaxSoldierAmount >= Course::ConstResourceMaps::UNIT_LIMITS) {
        newMaxSoldierAmount = Course::ConstResourceMaps::UNIT_LIMITS;
    }

    maxUnitAmount_ = newMaxUnitAmount;
    maxSoldierAmount_ = newMaxSoldierAmount;
}

void PlayerBase::eliminateExcessUnits() {
    updateUnitAmounts();
    while (getFreeUnitAmount() < 0) {
        int ind = 0;
        for (auto o : objects_) {
            if (std::dynamic_pointer_cast<Course::UnitBase>(o.lock()) != nullptr) {
                std::shared_ptr<Course::UnitBase> unit =
                        std::dynamic_pointer_cast<Course::UnitBase>(o.lock());
                if (unit->getType() == "BasicWorker" or unit->getType() == "Expert") {
                    objectManager_.lock()->getGameScene()->removeItem(unit);
                    unit->getParentTile()->removeUnit(unit);

                    break;
                }

            }
            ++ind;
        }
        updateUnitAmounts();
    }

    while (getFreeSoldierAmount() < 0) {
        int ind = 0;
        for (auto o : objects_) {
            if (std::dynamic_pointer_cast<Course::UnitBase>(o.lock()) != nullptr) {
                std::shared_ptr<Course::UnitBase> unit =
                        std::dynamic_pointer_cast<Course::UnitBase>(o.lock());
                if (unit->getType() == "Soldier") {
                    objectManager_.lock()->getGameScene()->removeItem(unit);
                    unit->getParentTile()->removeUnit(unit);
                    break;
                }
            }
            ++ind;
        }
        updateUnitAmounts();
    }
}

int PlayerBase::getCurrentUnitAmount()
{
    return getCurrentBasicWorkerAmount() + getCurrentExpertAmount();
}

int PlayerBase::getCurrentBasicWorkerAmount()
{
    int amount = 0;
    for (auto o : objects_) {
        if (std::dynamic_pointer_cast<Course::UnitBase>(o.lock()) != nullptr) {
            std::shared_ptr<Course::UnitBase> unit = std::dynamic_pointer_cast<Course::UnitBase>(o.lock());
            if (unit->getType() == "BasicWorker") {
                amount += 1;
            }

        }
    }
    return amount;
}

int PlayerBase::getCurrentExpertAmount()
{
    int amount = 0;
    for (auto o : objects_) {
        if (std::dynamic_pointer_cast<Course::UnitBase>(o.lock()) != nullptr) {
            std::shared_ptr<Course::UnitBase> unit =
                    std::dynamic_pointer_cast<Course::UnitBase>(o.lock());
            if (unit->getType() == "Expert") {
                amount += 1;
            }

        }
    }
    return amount;
}

int PlayerBase::getCurrentSoldierAmount()
{
    int amount = 0;
    for (auto o : objects_) {
        if (std::dynamic_pointer_cast<Course::UnitBase>(o.lock()) != nullptr) {
            std::shared_ptr<Course::UnitBase> unit = std::dynamic_pointer_cast<Course::UnitBase>(o.lock());
            if (unit->getType() == "Soldier") {
                amount += 1;
            }

        }
    }
    return amount;
}

void PlayerBase::limitResources()
{
    std::map<Course::BasicResource, int>::iterator it;

    for ( it = resources_.begin(); it != resources_.end(); it++ )
    {
        if (resources_.at(it->first) >= Course::ConstResourceMaps::RESOURCE_LIMITS.at(it->first)) {
            resources_[it->first] = Course::ConstResourceMaps::RESOURCE_LIMITS.at(it->first);
        }
    }
}



} // namespace Course

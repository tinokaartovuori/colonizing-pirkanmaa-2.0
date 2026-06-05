/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: menusceneitem.cpp, see menusceneitem.h for more info         #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "menusceneitem.h"

#include <QDebug>
#include <iostream>
#include <QFontDatabase>
#include <QTextDocument>
#include <QTextBlock>
#include <QAbstractTextDocumentLayout>

namespace Student {

MenuSceneItem::MenuSceneItem(const std::shared_ptr<Student::MenuObject> &obj,
                             std::shared_ptr<Student::MenuView>& mv):
    SceneItem(obj), m_upperLayer(mv)
{
    relative_coordinates = getBoundObject()->getCoordinate().asQpoint();
    absolute_coordinates = mv->getCoordinatePtr()->asQpoint();
    gridSize_ = mv->getGridSize();

}


MenuSceneItem::MenuSceneItem(const std::shared_ptr<MenuObject> &obj,
                             std::shared_ptr<MenuObjectContainer> &mv):
    SceneItem(obj), m_upperLayer(mv)
{
    relative_coordinates = getBoundObject()->getCoordinate().asQpoint();
    gridSize_ = mv->getGridSize();
    absolute_coordinates = mv->getAbsoluteCoordinates();

}


MenuSceneItem::MenuSceneItem(const std::shared_ptr<MenuObject> &obj):
    SceneItem(obj)
{
    relative_coordinates = getBoundObject()->getCoordinate().asQpoint();
    absolute_coordinates = QPoint(0, 0);
}

MenuSceneItem::MenuSceneItem(const std::shared_ptr<Student::MenuView> &obj):
    SceneItem(obj)
{
    // MenuView Coordinates
    absolute_coordinates = getBoundObject()->getCoordinate().asQpoint();
    relative_coordinates = QPoint(0, 0);
}

QRectF MenuSceneItem::boundingRect() const
{

    return QRectF(absolute_coordinates + relative_coordinates * gridSize_,
                  absolute_coordinates + relative_coordinates * gridSize_
                  + QPoint(width_ , height_));
}

std::string MenuSceneItem::getType()
{
    return "MenuSceneItem";
}


void MenuSceneItem::paint(QPainter *painter,
                    const QStyleOptionGraphicsItem *option,
                    QWidget *widget)
{
    Q_UNUSED( option ); Q_UNUSED( widget );

    setAcceptHoverEvents(true);

    // Multitile contruct drawing
    int index;
    if (std::dynamic_pointer_cast<Student::MenuObject>(baseObject_)->isMultiPixMap()) {
        if (!std::dynamic_pointer_cast<Student::MenuObject>(baseObject_)->isInverseMultiPixMap()) {
            for (int y = 0; y < baseObject_->getHeight(); ++y) {
                 for (int x = 0; x < baseObject_->getWidth(); ++x) {
                     if (y == 0 && x == 0) {
                         index = 0;
                     }
                     else if (y == 0 && x == baseObject_->getWidth() - 1) {
                         index = 1;
                     }
                     else if (y == baseObject_->getHeight() - 1 && x == baseObject_->getWidth() - 1) {
                         index = 2;
                     }
                     else if (y == baseObject_->getHeight() - 1 && x == 0) {
                         index = 3;
                     }
                     else if (y == 0) {
                         index = 7;
                     }
                     else if (y == baseObject_->getHeight() - 1) {
                         index = 5;
                     }
                     else if (x == 0) {
                         index = 4;
                     }
                     else if (x == baseObject_->getWidth() - 1) {
                         index = 6;
                     }
                     else {
                         index = 8;
                     }

                     painter->drawPixmap(absolute_coordinates.x() + ((relative_coordinates.x() + x) * gridSize_),
                                         absolute_coordinates.y() + ((relative_coordinates.y() + y) * gridSize_),
                                         gridSize_, gridSize_,
                                         itemPixmap_.at(index));
                 }
             }
        } else {
            for (int y = 0; y < baseObject_->getHeight(); ++y) {
                 for (int x = 0; x < baseObject_->getWidth(); ++x) {
                     if (y == 0 && x == 0) {
                         index = 2;
                     }
                     else if (y == 0 && x == baseObject_->getWidth() - 1) {
                         index = 3;
                     }
                     else if (y == baseObject_->getHeight() - 1 && x == baseObject_->getWidth() - 1) {
                         index = 0;
                     }
                     else if (y == baseObject_->getHeight() - 1 && x == 0) {
                         index = 1;
                     }
                     else if (y == 0) {
                         index = 5;
                     }
                     else if (y == baseObject_->getHeight() - 1) {
                         index = 7;
                     }
                     else if (x == 0) {
                         index = 6;
                     }
                     else if (x == baseObject_->getWidth() - 1) {
                         index = 4;
                     }
                     else {
                         index = 8;
                     }

                     painter->drawPixmap(absolute_coordinates.x() + ((relative_coordinates.x() + x) * gridSize_),
                                         absolute_coordinates.y() + ((relative_coordinates.y() + y) * gridSize_),
                                         gridSize_, gridSize_,
                                         itemPixmap_.at(index).transformed(transform().rotate(180)));
                 }
            }
        }

    }

    else if (itemPixmap_.size() > 0) {
        painter->drawPixmap(absolute_coordinates.x() + (relative_coordinates.x() * gridSize_),
                            absolute_coordinates.y() + (relative_coordinates.y() * gridSize_),
                            width_, height_,
                            itemPixmap_.at(currentImageFrame_ - 1));
    }

    if (baseObject_->getType() == "Label" || baseObject_->getType() == "Button") {
        std::shared_ptr<Student::iLabel> obj = std::dynamic_pointer_cast<Student::iLabel>(baseObject_);

        QFont game_font(QFontDatabase::applicationFontFamilies(0).at(0));

        /*painter->drawRect(QRect(absolute_coordinates.x() + (relative_coordinates.x() * gridSize_),
                                absolute_coordinates.y() + (relative_coordinates.y() * gridSize_),
                                width_, height_));*/

        game_font.setPixelSize(obj->getFontSize());
        painter->setFont(game_font);

        QTextDocument text;
        QAbstractTextDocumentLayout::PaintContext ctx;
        ctx.clip = QRectF(0, 0, width_, height_);
        std::string red = std::to_string(obj->getColor().red());
        std::string green = std::to_string(obj->getColor().green());
        std::string blue = std::to_string(obj->getColor().blue());

        if (obj->getStyle() == "LEFT") {
            QString lable_text = QString::fromStdString("<p style='line-height: 150%; color: rgb(" + red + "," + green + "," + blue + ")'>" + m_text + "</p>");
            text.setHtml(lable_text);
            text.setDefaultFont(game_font);

            text.setDocumentMargin(obj->getMargin());

            if (obj->noRightMargin()) {
                text.setTextWidth(width_ + obj->getMargin());
            } else {
                text.setTextWidth(width_);
            }

            painter->translate(QPointF(absolute_coordinates.x() + (relative_coordinates.x() * gridSize_),
                                       absolute_coordinates.y() + (relative_coordinates.y() * gridSize_ + obj->getOffset())));

        }
        else if (obj->getStyle() == "CENTER") {
            QString lable_text = QString::fromStdString("<p style='line-height: 150%; text-align: center; color: rgb(" + red + "," + green + "," + blue + ")'>" + m_text + "</p>");
            text.setHtml(lable_text);
            text.setDefaultFont(game_font);
            text.setDocumentMargin(obj->getMargin());
            text.setTextWidth(width_);
            int offset = height_ / 2 - text.documentLayout()->documentSize().height() / 2 + 4;
            painter->translate(QPointF(absolute_coordinates.x() + (relative_coordinates.x() * gridSize_),
                                       absolute_coordinates.y() + (relative_coordinates.y() * gridSize_ + offset + obj->getOffset())));

        }
        else if (obj->getStyle() == "LEFT-CENTER") {
            QString lable_text = QString::fromStdString("<p style='line-height: 150%; color: rgb(" + red + "," + green + "," + blue + ")'>" + m_text + "</p>");
            text.setHtml(lable_text);
            text.setDefaultFont(game_font);
            text.setDocumentMargin(obj->getMargin());
            text.setTextWidth(width_);
            int offset = height_ / 2 - text.documentLayout()->documentSize().height() / 2 + 4;
            painter->translate(QPointF(absolute_coordinates.x() + (relative_coordinates.x() * gridSize_),
                                       absolute_coordinates.y() + (relative_coordinates.y() * gridSize_ + offset + obj->getOffset())));

        }
        else if (obj->getStyle() == "VERTICAL-CENTER") {
            QString lable_text = QString::fromStdString("<p style='line-height: 150%; text-align: center; color: rgb(" + red + "," + green + "," + blue + ")'>" + m_text + "</p>");
            text.setHtml(lable_text);
            text.setDefaultFont(game_font);

            text.setDocumentMargin(obj->getMargin());

            if (obj->noRightMargin()) {
                text.setTextWidth(width_ + obj->getMargin());
            } else {
                text.setTextWidth(width_);
            }

            painter->translate(QPointF(absolute_coordinates.x() + (relative_coordinates.x() * gridSize_),
                                       absolute_coordinates.y() + (relative_coordinates.y() * gridSize_ + obj->getOffset())));

        }

        text.documentLayout()->draw(painter, ctx);

    }

}

void MenuSceneItem::setText()
{
    if (std::dynamic_pointer_cast<Student::iLabel>(baseObject_) != nullptr) {
        std::shared_ptr<Student::iLabel> obj = std::dynamic_pointer_cast<Student::iLabel>(baseObject_);
        m_text = obj->getText();
        m_fontSize = obj->getFontSize();
        m_color = obj->getColor();
        m_style = obj->getStyle();
    }
}


void MenuSceneItem::updateLoc()
{
    if ( !baseObject_ )
    {
        delete this;
    }
    else {
        relative_coordinates = baseObject_->getCoordinate().asQpoint();

    }
}

void MenuSceneItem::setUpperLayer(std::shared_ptr<Student::MenuView> mv) {
    m_upperLayer = mv;
    absolute_coordinates = mv->getCoordinatePtr()->asQpoint();
}

void MenuSceneItem::setUpperLayer(std::shared_ptr<Student::MenuObjectContainer> mv) {
    m_upperLayer = mv;
    relative_coordinates = mv->getCoordinatePtr()->asQpoint();
}

} //namespace Course



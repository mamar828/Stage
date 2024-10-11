#include <time.h>
#include "time.h"
#include <iostream>
#include <sstream>

using namespace std;

/**
 * \brief Create a Time object
 * \return A Time Object
 */
Time::Time() :
m_hours(0), m_minutes(0), m_seconds(0) {};

/**
 * \brief Create a Time object
 * \param[in] p_hours Number of hours
 * \param[in] p_minutes Number of minutes
 * \param[in] p_seconds Number of seconds
 * \return Time object
 */
Time::Time(const int p_hours, const int p_minutes, const int p_seconds) :
m_hours(p_hours), m_minutes(p_minutes), m_seconds(p_seconds) {};

/**
 * \brief Set the object to the current time.
 */
void Time::set_to_current_time()
{
    int seconds = time(NULL);
    int gmt_hours = (seconds / 3600) % 24;
    m_hours = gmt_hours - 4;
    if (m_hours < 0)
    m_hours += 24;
    m_minutes = (seconds / 60) % 60;
    m_seconds = seconds % 60;
}

/**
 * \brief Assign a number of hours
 * \param[in] p_hours Number of hours to assign
 */
void Time::asg_hours(const int p_hours)
{
    m_hours = (p_hours >= 0 && p_hours < 24) ? p_hours : 0;
}

/**
 * \brief Assign a number of minutes
 * \param[in] p_minutes Number of minutes to assign
 */
void Time::asg_minutes(const int p_minutes)
{
    m_minutes = (p_minutes >= 0 && p_minutes < 60) ? p_minutes : 0;
}

/**
 * \brief Assign a number of seconds
 * \param[in] p_seconds Number of seconds to assign
 */
void Time::asg_seconds(const int p_seconds)
{
    m_seconds = (p_seconds >= 0 && p_seconds < 60) ? p_seconds : 0;
}

/**
 * \brief Get the number of hours
 * \return m_hours Number of hours
 */
const int Time::get_hours() const
{
    return m_hours;
}

/**
 * \brief Get the number of minutes
 * \return m_minutes Number of minutes
 */
const int Time::get_minutes() const
{
    return m_minutes;
}

/**
 * \brief Get the number of seconds
 * \return m_seconds Number of seconds
 */
const int Time::get_seconds() const
{
    return m_seconds;
}

string format(int qty)
{
    ostringstream stream;
    if (qty < 10)
    stream << '0';
    stream << qty;
    return stream.str();
}

string Time::get_formatted_time()
{
    ostringstream oss;
    set_to_current_time();
    oss << format(m_hours) << ':' << format(m_minutes) << ':' << format(m_seconds);
    return oss.str();
}

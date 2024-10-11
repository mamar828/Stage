#include <string>

class Time
{
    public:
        Time();
        Time(const int p_hours, const int p_minutes, const int p_seconds);
        void set_to_current_time();
        void asg_hours(const int p_hours);
        void asg_minutes(const int p_minutes);
        void asg_seconds(const int p_seconds);
        const int get_hours() const;
        const int get_minutes() const;
        const int get_seconds() const;
        std::string get_formatted_time();
    private:
        int m_hours;
        int m_minutes;
        int m_seconds;
};

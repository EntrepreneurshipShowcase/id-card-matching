from flask import Blueprint, jsonify

def get_blueprint(database):

    admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

    @admin_bp.route("/student_attendance/<int:id>")
    def get_attendance(id):
        attendance_status = database.get_attendance(id)
        return attendance_status
    @admin_bp.route("/end_day")
    def end_day():
        database.end_day()
    @admin_bp.route("/update_attendance/<int:id>/<bool:status>")
    def update_attendance(id, status):
        database.update_attendance(id, status)
    @admin_bp.route("/get_all_attendance/")
    def get_all_attendance():
        attendance_dict = database.get_all_attendance().to_dict()
        return attendance_dict
    return admin_bp
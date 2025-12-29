import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
import json
import os
from datetime import date, datetime, timedelta
import calendar as cal_module

# --- CONFIGURATION ---
CONFIG = {
    "PAI_TOTAL_PAGES": 417,
    "DL_TOTAL_PAGES": 199,
    "PAI_EXAM_DATE": date(2026, 1, 19),
    "DL_EXAM_DATE": date(2026, 2, 6),
    "READING_DEADLINE": date(2026, 1, 7),
    "DATA_FILE": "study_data_gui.json"
}

# --- DEFAULT DATA ---
DEFAULT_STATE = {
    "pai_pages_read": 106,
    "dl_pages_read": 35,
    "dl_project_pct": 35,
    "aise_project_pct": 0,
    "tasks": [
        {"text": "Email group about report writing", "done": False, "deadline": None},
        {"text": "Create Mind Map template", "done": False, "deadline": None}
    ]
}

class StudyApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("🎓 Uni Study Commander")
        self.geometry("900x750")
        self.resizable(True, True)
        self.minsize(800, 700)
        
        self.data = self.load_data()
        self.create_widgets()
        self.update_dashboard()
        self.update_calendar_events()

    def load_data(self):
        if not os.path.exists(CONFIG["DATA_FILE"]):
            return DEFAULT_STATE
        try:
            with open(CONFIG["DATA_FILE"], 'r') as f:
                return json.load(f)
        except:
            return DEFAULT_STATE

    def save_data(self):
        with open(CONFIG["DATA_FILE"], 'w') as f:
            json.dump(self.data, f, indent=4)

    def create_widgets(self):
        # -- MAIN TABS --
        self.notebook = ttk.Notebook(self, bootstyle="primary")
        self.notebook.pack(pady=10, expand=True, fill='both', padx=10)

        self.tab_dashboard = ttk.Frame(self.notebook)
        self.tab_todo = ttk.Frame(self.notebook)
        self.tab_update = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_dashboard, text='  📊 Dashboard  ')
        self.notebook.add(self.tab_todo, text='  ✅ To-Do List  ')
        self.notebook.add(self.tab_update, text='  📝 Update Progress  ')

        self.build_dashboard_tab()
        self.build_todo_tab()
        self.build_update_tab()

    # ================= DASHBOARD TAB =================
    def build_dashboard_tab(self):
        # Create a scrollable frame for dashboard
        main_container = ttk.Frame(self.tab_dashboard)
        main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left column - Stats
        left_col = ttk.Frame(main_container)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Right column - Calendar
        right_col = ttk.Frame(main_container)
        right_col.pack(side="right", fill="both", padx=(5, 0))

        # ---- LEFT COLUMN ----
        # Frame: Countdowns
        frame_time = ttk.Labelframe(left_col, text="⏳ Countdowns", bootstyle="info")
        frame_time.pack(padx=10, pady=5, fill="x")
        
        self.lbl_pai_days = ttk.Label(frame_time, text="", font=("Segoe UI", 11))
        self.lbl_pai_days.pack(anchor="w", padx=15, pady=3)
        
        self.lbl_dl_days = ttk.Label(frame_time, text="", font=("Segoe UI", 11))
        self.lbl_dl_days.pack(anchor="w", padx=15, pady=3)

        self.lbl_read_deadline = ttk.Label(frame_time, text="", font=("Segoe UI", 10, "bold"), bootstyle="danger")
        self.lbl_read_deadline.pack(anchor="w", padx=15, pady=5)

        # Frame: Reading Goals
        frame_read = ttk.Labelframe(left_col, text="📚 Script Reading Progress", bootstyle="success")
        frame_read.pack(padx=10, pady=5, fill="x")

        # PAI
        pai_frame = ttk.Frame(frame_read)
        pai_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(pai_frame, text="Probabilistic AI", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.prog_pai = ttk.Progressbar(pai_frame, orient="horizontal", length=100, mode="determinate", bootstyle="success-striped")
        self.prog_pai.pack(fill="x", pady=2)
        self.lbl_pai_stats = ttk.Label(pai_frame, text="0/0 pages", font=("Segoe UI", 9))
        self.lbl_pai_stats.pack(anchor="e")

        # DL
        dl_frame = ttk.Frame(frame_read)
        dl_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(dl_frame, text="Deep Learning", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.prog_dl = ttk.Progressbar(dl_frame, orient="horizontal", length=100, mode="determinate", bootstyle="info-striped")
        self.prog_dl.pack(fill="x", pady=2)
        self.lbl_dl_stats = ttk.Label(dl_frame, text="0/0 pages", font=("Segoe UI", 9))
        self.lbl_dl_stats.pack(anchor="e")

        # Daily Target Box
        self.lbl_daily_target = ttk.Label(frame_read, text="", font=("Segoe UI", 11, "bold"), 
                                          bootstyle="inverse-warning", padding=10)
        self.lbl_daily_target.pack(fill="x", padx=10, pady=10)

        # Frame: Projects
        frame_proj = ttk.Labelframe(left_col, text="🛠️ Project Completion", bootstyle="warning")
        frame_proj.pack(padx=10, pady=5, fill="x")

        # DL Project
        proj_dl_frame = ttk.Frame(frame_proj)
        proj_dl_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(proj_dl_frame, text="Deep Learning Project", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.prog_proj_dl = ttk.Progressbar(proj_dl_frame, orient="horizontal", length=100, mode="determinate", bootstyle="warning-striped")
        self.prog_proj_dl.pack(fill="x", pady=2)
        self.lbl_proj_dl_stats = ttk.Label(proj_dl_frame, text="0%", font=("Segoe UI", 9))
        self.lbl_proj_dl_stats.pack(anchor="e")

        # AISE Project
        proj_aise_frame = ttk.Frame(frame_proj)
        proj_aise_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(proj_aise_frame, text="AI in Sci/Eng Project", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.prog_proj_aise = ttk.Progressbar(proj_aise_frame, orient="horizontal", length=100, mode="determinate", bootstyle="danger-striped")
        self.prog_proj_aise.pack(fill="x", pady=2)
        self.lbl_proj_aise_stats = ttk.Label(proj_aise_frame, text="0%", font=("Segoe UI", 9))
        self.lbl_proj_aise_stats.pack(anchor="e")

        # ---- RIGHT COLUMN - Calendar ----
        frame_cal = ttk.Labelframe(right_col, text="📅 Calendar & Events", bootstyle="primary")
        frame_cal.pack(padx=5, pady=5, fill="both", expand=True)
        
        # Custom Calendar View
        self.current_month = date.today().month
        self.current_year = date.today().year
        
        # Month navigation
        nav_frame = ttk.Frame(frame_cal)
        nav_frame.pack(fill="x", padx=10, pady=5)
        
        btn_prev = ttk.Button(nav_frame, text="◀", width=3, command=self.prev_month, bootstyle="outline")
        btn_prev.pack(side="left")
        
        self.month_label = ttk.Label(nav_frame, text="", font=("Segoe UI", 12, "bold"))
        self.month_label.pack(side="left", expand=True)
        
        btn_next = ttk.Button(nav_frame, text="▶", width=3, command=self.next_month, bootstyle="outline")
        btn_next.pack(side="right")
        
        # Calendar grid
        self.cal_frame = ttk.Frame(frame_cal)
        self.cal_frame.pack(fill="x", padx=10, pady=5)
        
        # Events list
        events_frame = ttk.Frame(frame_cal)
        events_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ttk.Label(events_frame, text="📌 Upcoming Events", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Scrollable events list
        self.events_text = tk.Text(events_frame, height=10, width=30, font=("Segoe UI", 9),
                                   bg='#2b3e50', fg='white', relief="flat", padx=10, pady=5)
        self.events_text.pack(fill="both", expand=True)
        self.events_text.config(state="disabled")
    
    def prev_month(self):
        if self.current_month == 1:
            self.current_month = 12
            self.current_year -= 1
        else:
            self.current_month -= 1
        self.update_calendar_view()
    
    def next_month(self):
        if self.current_month == 12:
            self.current_month = 1
            self.current_year += 1
        else:
            self.current_month += 1
        self.update_calendar_view()
    
    def update_calendar_view(self):
        # Clear calendar grid
        for widget in self.cal_frame.winfo_children():
            widget.destroy()
        
        # Update month label
        month_name = cal_module.month_name[self.current_month]
        self.month_label.config(text=f"{month_name} {self.current_year}")
        
        # Get important dates
        important_dates = self.get_important_dates()
        
        # Day headers
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, day in enumerate(days):
            lbl = ttk.Label(self.cal_frame, text=day, font=("Segoe UI", 8, "bold"), width=4)
            lbl.grid(row=0, column=i, padx=1, pady=2)
        
        # Get calendar for month
        cal = cal_module.Calendar(firstweekday=0)
        month_days = cal.monthdayscalendar(self.current_year, self.current_month)
        today = date.today()
        
        for week_num, week in enumerate(month_days):
            for day_num, day in enumerate(week):
                if day == 0:
                    lbl = ttk.Label(self.cal_frame, text="", width=4)
                else:
                    current_date = date(self.current_year, self.current_month, day)
                    style = "secondary"
                    
                    # Check if it's today
                    if current_date == today:
                        style = "info"
                    # Check if it has an event
                    elif current_date in important_dates:
                        event_type = important_dates[current_date]
                        if event_type == "exam":
                            style = "danger"
                        elif event_type == "deadline":
                            style = "warning"
                        elif event_type == "task":
                            style = "primary"
                    
                    lbl = ttk.Label(self.cal_frame, text=str(day), width=4, bootstyle=style, 
                                   font=("Segoe UI", 9), anchor="center")
                lbl.grid(row=week_num + 1, column=day_num, padx=1, pady=1)
    
    def get_important_dates(self):
        """Return dict of important dates and their types"""
        dates = {}
        
        # Add exam dates
        dates[CONFIG["PAI_EXAM_DATE"]] = "exam"
        dates[CONFIG["DL_EXAM_DATE"]] = "exam"
        dates[CONFIG["READING_DEADLINE"]] = "deadline"
        
        # Add task deadlines
        for task in self.data["tasks"]:
            if task.get("deadline") and not task["done"]:
                try:
                    deadline_date = datetime.strptime(task["deadline"], "%Y-%m-%d").date()
                    if deadline_date not in dates:  # Don't override exams
                        dates[deadline_date] = "task"
                except ValueError:
                    pass
        
        return dates

    # ================= TO-DO TAB =================
    def build_todo_tab(self):
        # Header with title
        header_frame = ttk.Frame(self.tab_todo)
        header_frame.pack(pady=10, padx=20, fill="x")
        ttk.Label(header_frame, text="📋 Manage Your Tasks", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        
        # Input frame with better layout
        input_frame = ttk.Labelframe(self.tab_todo, text="Add New Task", bootstyle="info")
        input_frame.pack(pady=10, padx=20, fill="x")
        
        # Row 1: Task input
        row1 = ttk.Frame(input_frame)
        row1.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(row1, text="Task:", font=("Segoe UI", 10)).pack(side="left", padx=(0, 5))
        self.ent_task = ttk.Entry(row1, width=50, font=("Segoe UI", 10))
        self.ent_task.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.ent_task.bind("<Return>", lambda e: self.add_task())
        
        # Row 2: Deadline and buttons
        row2 = ttk.Frame(input_frame)
        row2.pack(fill="x", padx=10, pady=(0, 10))
        
        ttk.Label(row2, text="Deadline:", font=("Segoe UI", 10)).pack(side="left", padx=(0, 5))
        self.deadline_entry = ttk.DateEntry(row2, bootstyle="info", dateformat="%Y-%m-%d")
        self.deadline_entry.pack(side="left", padx=(0, 10))
        
        # Checkbox for using deadline
        self._use_deadline = tk.BooleanVar(value=True)
        self.chk_deadline = ttk.Checkbutton(row2, text="Set deadline", variable=self._use_deadline, 
                                             bootstyle="round-toggle")
        self.chk_deadline.pack(side="left", padx=(0, 10))
        
        btn_add = ttk.Button(row2, text="➕ Add Task", command=self.add_task, bootstyle="success")
        btn_add.pack(side="right")

        # Listbox area with scrollbar
        list_container = ttk.Labelframe(self.tab_todo, text="Your Tasks", bootstyle="secondary")
        list_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Canvas for scrolling
        canvas = tk.Canvas(list_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview, bootstyle="round")
        self.list_frame = ttk.Frame(canvas)
        
        self.list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y", pady=5)
        
        # Mouse wheel scrolling
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        self.task_vars = []
        self.refresh_tasks_ui()

    def refresh_tasks_ui(self):
        # Clear existing widgets in list frame
        for widget in self.list_frame.winfo_children():
            widget.destroy()
        
        self.task_vars = []
        today = date.today()
        
        if not self.data["tasks"]:
            ttk.Label(self.list_frame, text="No tasks yet! Add one above.", 
                     font=("Segoe UI", 11), bootstyle="secondary").pack(pady=20)
            return
        
        for idx, task in enumerate(self.data["tasks"]):
            var = tk.BooleanVar(value=task["done"])
            self.task_vars.append(var)
            
            # Task card
            f = ttk.Frame(self.list_frame, bootstyle="default")
            f.pack(fill="x", pady=3, padx=5)
            
            # Left side - checkbox and text
            left_frame = ttk.Frame(f)
            left_frame.pack(side="left", fill="x", expand=True)
            
            style = "success-round-toggle" if task["done"] else "round-toggle"
            chk = ttk.Checkbutton(left_frame, text=task["text"], variable=var, 
                                  command=lambda i=idx: self.toggle_task(i),
                                  bootstyle=style)
            chk.pack(side="left", padx=5)
            
            # Display deadline if present
            deadline = task.get("deadline")
            if deadline:
                try:
                    deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
                    days_until = (deadline_date - today).days
                    
                    # Color code based on urgency
                    if task["done"]:
                        style = "success"
                        text = f"✓ Completed"
                    elif days_until < 0:
                        style = "danger"
                        text = f"⚠ OVERDUE by {abs(days_until)} days"
                    elif days_until == 0:
                        style = "warning"
                        text = f"⏰ DUE TODAY"
                    elif days_until <= 3:
                        style = "warning"
                        text = f"⚠ {days_until} days left"
                    else:
                        style = "info"
                        text = f"📅 {deadline}"
                    
                    lbl_deadline = ttk.Label(left_frame, text=text, font=("Segoe UI", 9), bootstyle=style)
                    lbl_deadline.pack(side="left", padx=(15, 0))
                except ValueError:
                    pass  # Invalid date format
            
            btn_del = ttk.Button(f, text="🗑", width=3, command=lambda i=idx: self.delete_task(i),
                                bootstyle="danger-outline")
            btn_del.pack(side="right", padx=5)
        
        # Update calendar view when tasks change
        if hasattr(self, 'cal_frame'):
            self.update_calendar_view()
            self.update_calendar_events()

    def add_task(self):
        text = self.ent_task.get().strip()
        if text:
            # Get deadline if checkbox is checked
            deadline = None
            if hasattr(self, '_use_deadline') and isinstance(self._use_deadline, tk.BooleanVar):
                if self._use_deadline.get():
                    # ttkbootstrap DateEntry returns date from entry.entry.get()
                    deadline = self.deadline_entry.entry.get()
            
            self.data["tasks"].append({"text": text, "done": False, "deadline": deadline})
            self.ent_task.delete(0, tk.END)
            self.save_data()
            self.refresh_tasks_ui()
            Messagebox.show_info("Task added successfully!", "Task Added")
    
    def clear_deadline(self):
        """Mark that no deadline should be used for the next task"""
        pass  # No longer needed with checkbox approach

    def toggle_task(self, index):
        # Update data from checkbox state
        self.data["tasks"][index]["done"] = self.task_vars[index].get()
        self.save_data()
        self.refresh_tasks_ui()

    def delete_task(self, index):
        del self.data["tasks"][index]
        self.save_data()
        self.refresh_tasks_ui()

    # ================= UPDATE TAB =================
    def build_update_tab(self):
        # Header
        header_frame = ttk.Frame(self.tab_update)
        header_frame.pack(pady=20, padx=30, fill="x")
        ttk.Label(header_frame, text="📝 Update Your Progress", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(header_frame, text="Keep track of your study progress", 
                 font=("Segoe UI", 10), bootstyle="secondary").pack(anchor="w")
        
        # Main form
        frame = ttk.Labelframe(self.tab_update, text="Progress Updates", bootstyle="primary")
        frame.pack(padx=50, pady=20, fill="both", expand=True)

        # PAI Pages
        pai_frame = ttk.Frame(frame)
        pai_frame.pack(fill="x", padx=20, pady=15)
        ttk.Label(pai_frame, text="📖 Current PAI Page Number:", font=("Segoe UI", 11)).pack(anchor="w")
        self.ent_pai_pg = ttk.Entry(pai_frame, font=("Segoe UI", 11))
        self.ent_pai_pg.insert(0, str(self.data["pai_pages_read"]))
        self.ent_pai_pg.pack(fill="x", pady=(5, 0))

        # DL Pages
        dl_frame = ttk.Frame(frame)
        dl_frame.pack(fill="x", padx=20, pady=15)
        ttk.Label(dl_frame, text="📖 Current DL Page Number:", font=("Segoe UI", 11)).pack(anchor="w")
        self.ent_dl_pg = ttk.Entry(dl_frame, font=("Segoe UI", 11))
        self.ent_dl_pg.insert(0, str(self.data["dl_pages_read"]))
        self.ent_dl_pg.pack(fill="x", pady=(5, 0))

        # DL Project
        dl_proj_frame = ttk.Frame(frame)
        dl_proj_frame.pack(fill="x", padx=20, pady=15)
        ttk.Label(dl_proj_frame, text="🛠️ DL Project Completion (%):", font=("Segoe UI", 11)).pack(anchor="w")
        self.ent_dl_pct = ttk.Entry(dl_proj_frame, font=("Segoe UI", 11))
        self.ent_dl_pct.insert(0, str(self.data["dl_project_pct"]))
        self.ent_dl_pct.pack(fill="x", pady=(5, 0))

        # AISE Project
        aise_frame = ttk.Frame(frame)
        aise_frame.pack(fill="x", padx=20, pady=15)
        ttk.Label(aise_frame, text="🛠️ AISE Project Completion (%):", font=("Segoe UI", 11)).pack(anchor="w")
        self.ent_aise_pct = ttk.Entry(aise_frame, font=("Segoe UI", 11))
        self.ent_aise_pct.insert(0, str(self.data["aise_project_pct"]))
        self.ent_aise_pct.pack(fill="x", pady=(5, 0))

        # Save button
        btn_save = ttk.Button(frame, text="💾 Save Updates", command=self.process_update, 
                             bootstyle="success", padding=15)
        btn_save.pack(fill="x", padx=20, pady=25)

    def process_update(self):
        try:
            self.data["pai_pages_read"] = int(self.ent_pai_pg.get())
            self.data["dl_pages_read"] = int(self.ent_dl_pg.get())
            self.data["dl_project_pct"] = int(self.ent_dl_pct.get())
            self.data["aise_project_pct"] = int(self.ent_aise_pct.get())
            
            self.save_data()
            self.update_dashboard()
            Messagebox.show_info("Progress Updated Successfully!", "Success")
            self.notebook.select(self.tab_dashboard)
        except ValueError:
            Messagebox.show_error("Please enter valid numbers only.", "Error")

    # ================= CALENDAR EVENTS =================
    def update_calendar_events(self):
        """Update the events list widget"""
        today = date.today()
        events = []
        
        # Add exam dates
        pai_exam = CONFIG["PAI_EXAM_DATE"]
        dl_exam = CONFIG["DL_EXAM_DATE"]
        reading_deadline = CONFIG["READING_DEADLINE"]
        
        events.append((pai_exam, "📝 PAI Exam", "exam"))
        events.append((dl_exam, "📝 DL Exam", "exam"))
        events.append((reading_deadline, "📚 Reading Deadline", "deadline"))
        
        # Add task deadlines
        for task in self.data["tasks"]:
            if task.get("deadline") and not task["done"]:
                try:
                    deadline_date = datetime.strptime(task["deadline"], "%Y-%m-%d").date()
                    tag = "overdue" if deadline_date < today else "task"
                    events.append((deadline_date, f"📌 {task['text']}", tag))
                except ValueError:
                    pass
        
        # Update events text widget
        events.sort(key=lambda x: x[0])
        
        self.events_text.config(state="normal")
        self.events_text.delete(1.0, tk.END)
        
        # Show only upcoming events (next 60 days)
        upcoming = [e for e in events if e[0] >= today and e[0] <= today + timedelta(days=60)]
        
        if not upcoming:
            self.events_text.insert(tk.END, "No upcoming events in the next 60 days.\n")
        else:
            for evt_date, evt_text, evt_type in upcoming:
                days_until = (evt_date - today).days
                if days_until == 0:
                    day_str = "TODAY"
                elif days_until == 1:
                    day_str = "Tomorrow"
                else:
                    day_str = f"In {days_until} days"
                
                # Add color coding
                if evt_type == "exam":
                    prefix = "🔴"
                elif evt_type == "deadline":
                    prefix = "🟠"
                elif evt_type == "overdue":
                    prefix = "⚠️"
                else:
                    prefix = "🔵"
                
                self.events_text.insert(tk.END, f"{prefix} {evt_date.strftime('%b %d')} - {day_str}\n")
                self.events_text.insert(tk.END, f"   {evt_text}\n\n")
        
        self.events_text.config(state="disabled")

    # ================= LOGIC =================
    def update_dashboard(self):
        today = date.today()
        
        # Calc Days
        days_pai = (CONFIG["PAI_EXAM_DATE"] - today).days
        days_dl = (CONFIG["DL_EXAM_DATE"] - today).days
        days_read = (CONFIG["READING_DEADLINE"] - today).days
        
        self.lbl_pai_days.config(text=f"📝 PAI Exam: {days_pai} days remaining")
        self.lbl_dl_days.config(text=f"📝 DL Exam: {days_dl} days remaining")
        self.lbl_read_deadline.config(text=f"📚 Reading Phase Ends: {days_read} days left (Jan 7)")

        # Calc Pacing
        pai_rem = CONFIG["PAI_TOTAL_PAGES"] - self.data["pai_pages_read"]
        dl_rem = CONFIG["DL_TOTAL_PAGES"] - self.data["dl_pages_read"]
        
        eff_days = max(1, days_read)
        pai_pace = round(pai_rem / eff_days) if days_read > 0 else 0
        dl_pace = round(dl_rem / eff_days) if days_read > 0 else 0

        if days_read > 0:
            target_text = f"🎯 TODAY'S GOAL: Read {pai_pace} PAI pages & {dl_pace} DL pages"
        else:
            target_text = "⚠️ Reading Deadline Passed!"
        
        self.lbl_daily_target.config(text=target_text)

        # Update Progress Bars
        self.prog_pai["maximum"] = CONFIG["PAI_TOTAL_PAGES"]
        self.prog_pai["value"] = self.data["pai_pages_read"]
        pct_pai = (self.data["pai_pages_read"] / CONFIG["PAI_TOTAL_PAGES"]) * 100
        self.lbl_pai_stats.config(text=f"{self.data['pai_pages_read']}/{CONFIG['PAI_TOTAL_PAGES']} ({pct_pai:.1f}%)")

        self.prog_dl["maximum"] = CONFIG["DL_TOTAL_PAGES"]
        self.prog_dl["value"] = self.data["dl_pages_read"]
        pct_dl = (self.data["dl_pages_read"] / CONFIG["DL_TOTAL_PAGES"]) * 100
        self.lbl_dl_stats.config(text=f"{self.data['dl_pages_read']}/{CONFIG['DL_TOTAL_PAGES']} ({pct_dl:.1f}%)")

        self.prog_proj_dl["value"] = self.data["dl_project_pct"]
        self.lbl_proj_dl_stats.config(text=f"{self.data['dl_project_pct']}%")

        self.prog_proj_aise["value"] = self.data["aise_project_pct"]
        self.lbl_proj_aise_stats.config(text=f"{self.data['aise_project_pct']}%")
        
        # Update calendar view and events
        if hasattr(self, 'cal_frame'):
            self.update_calendar_view()
            self.update_calendar_events()

if __name__ == "__main__":
    app = StudyApp()
    app.mainloop()
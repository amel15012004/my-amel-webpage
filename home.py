import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import base64
import sqlite3
import os
import shutil
import hashlib
import contextlib
import resend
from ultralytics import YOLO
import cv2
import numpy as np
from openai import OpenAI
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# =======================
# 🔑 model
# =======================
model = YOLO("best.pt")
client = OpenAI(api_key="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")


# =======================
# 🌐 TRANSLATIONS
# =======================
TRANSLATIONS = {
    "EN": {
        # Navbar
        "nav_home": "Home", "nav_contact": "Contact", "nav_login": "Login", "nav_signup": "Sign Up",
        # Sidebar
        "sidebar_clinic": "🏥 DentAI Clinic", "sidebar_logout": "🚪 Logout",
        "menu_dashboard": "📊 Dashboard", "menu_diagnosis": "🧠 New Diagnosis",
        "menu_historic": "📁 Historic Reports", "menu_patients": "👨‍⚕️ Patients",
        "menu_profile": "👨‍⚕️ Profile ", "menu_settings": "⚙️ Settings",
        # Logout confirm
        "logout_confirm": "Are you sure you want to logout?",
        "logout_yes": "✅ Yes, Logout", "logout_cancel": "❌ Cancel",
        # Dashboard
        "dash_title": "📊 Clinic Overview", "dash_welcome": "Welcome Dr.",
        "dash_total_reports": "🧾 Total Reports", "dash_patients": "👤 Patients", "dash_today": "📅 Today",
        "dash_chart1": "🏆 Patients by Report Count", "dash_chart2": "📈 Patient Growth Over Time",
        "dash_no_patients": "No patient data yet", "dash_no_growth": "No growth data yet",
        "dash_recent": "🕒 Recent Reports", "dash_no_recent": "No recent reports",
        "dash_reports_per_patient": "Reports per Patient", "dash_num_reports": "Number of Reports",
        "dash_cumulative": "Cumulative Patient Growth", "dash_month": "Month", "dash_total_p": "Total Patients",
        # Diagnosis
        "diag_header": "🧠 AI Dental Diagnosis", "diag_upload_label": "Upload X-ray",
        "diag_detect": "🔍 Detect", "diag_analyse": "🧠 Analyse",
        "diag_detect_first": "⚠️ Please run **🔍 Detect** first before analysing.",
        "diag_generating": "Generating AI diagnosis...",
        "diag_num_detections": "Number of detections:",
        "diag_region": "Region", "diag_confidence": "Confidence",
        "diag_generated": "✅ AI Diagnosis Generated",
        "diag_patient_info": "👤 Patient Information",
        "diag_patient_name": "Patient Name", "diag_patient_age": "Patient Age",
        "diag_save_pdf": "📄 Save Report as PDF", "diag_fill_patient": "Please fill patient info",
        "diag_download": "⬇️ Download Report", "diag_success": "Report generated successfully!",
        # Historic
        "hist_header": "📁 Historic Reports", "hist_no_reports": "No reports yet",
        "hist_patient": "Patient", "hist_date": "Date", "hist_xray": "Original X-ray",
        # Patients
        "pat_header": "👨‍⚕️ Patients Folder", "pat_search": "🔍 Search patient by name",
        "pat_no_found": "No patients found", "pat_no_reports": "No reports for this patient",
        "pat_view": "👁️ View", "pat_delete_all": "❌ Delete ALL reports for",
        "pat_deleted": "Report deleted", "pat_all_deleted": "All reports for",
        "pat_all_deleted2": "deleted", "pat_download": "⬇️ Download PDF",
        "pat_close": "❎ Close Viewer",
        # Profile
        "prof_header": "👨‍⚕️ Profile Settings", "prof_no_photo": "No profile photo uploaded",
        "prof_name": "Full Name", "prof_email": "Email", "prof_dob": "Date of Birth",
        "prof_upload_photo": "Upload Profile Photo", "prof_save": "💾 Update Profile",
        "prof_updated": "✅ Profile updated successfully!", "prof_not_found": "User not found — please log out and log in again.",
        "prof_go_login": "🚪 Go to Login",
        # Settings
        "set_header": "⚙️ Advanced Settings", "set_appearance": "🎨 Appearance",
        "set_dark_mode": "🌙 Dark Mode", "set_language": "🌐 Language",
        "set_select_lang": "Select Language", "set_notifications": "🔔 Notifications",
        "set_email_notif": "📧 Email Notifications", "set_recent_notif": "Recent notifications:",
        "set_test_notif": "🧪 Add Test Notification", "set_security": "🔐 Security",
        "set_current_pass": "Current Password", "set_new_pass": "New Password",
        "set_change_pass": "🔑 Change Password", "set_pass_fill": "Please fill both password fields",
        "set_pass_updated": "Password updated!", "set_wrong_pass": "Wrong current password",
        "set_2fa": "Enable 2FA (Demo)", "set_otp_label": "Demo OTP:",
        "set_otp_input": "Enter OTP", "set_verify_otp": "Verify OTP",
        "set_otp_ok": "2FA Verified!", "set_otp_fail": "Invalid OTP",
        "set_save": "💾 Save All Settings", "set_saved": "Settings saved successfully!",
        "set_not_logged": "You are not logged in",
        # Pricing / Subscription
        "nav_pricing": "Pricing",
        "pricing_title": "💎 Choose Your Plan",
        "pricing_subtitle": "Start free, upgrade anytime. No hidden fees.",
        "plan_free_title": "🆓 Free Trial",
        "plan_free_price": "0 DA",
        "plan_free_period": "3 days",
        "plan_free_desc": "Try all features free for 3 days after account approval.",
        "plan_monthly_title": "📅 Monthly",
        "plan_monthly_price": "1 900 DA",
        "plan_monthly_period": "/ month",
        "plan_monthly_desc": "Full access to all features, billed monthly.",
        "plan_yearly_title": "🏆 Yearly",
        "plan_yearly_price": "18 000 DA",
        "plan_yearly_period": "/ year",
        "plan_yearly_desc": "Best value! Save 3 600 DA compared to monthly.",
        "plan_btn_select": "Select Plan",
        "plan_features": "✔ AI Diagnosis  ✔ PDF Reports  ✔ Patient Management  ✔ Dashboard",
        "plan_current": "✅ Current Plan",
        "payment_title": "💳 Payment – Manual Transfer",
        "payment_info": "To activate your subscription, please transfer the amount to the following account and then fill in the form below.",
        "payment_bank": "🏦 Bank: CPA (Crédit Populaire d'Algérie)",
        "payment_rib": "🔢 RIB: 00799999000123456789 12",
        "payment_name": "👤 Account Holder: DentAI SAS",
        "payment_ref_label": "Payment Reference (your email)",
        "payment_amount_label": "Amount Paid (DA)",
        "payment_proof_label": "Upload Payment Proof (screenshot / receipt)",
        "payment_submit": "📨 Submit Payment",
        "payment_success": "✅ Payment submitted! Your account will be activated within 24 hours.",
        "payment_fill": "Please fill all fields and upload proof.",
        "sub_expired_title": "⏳ Your subscription has expired",
        "sub_expired_msg": "Please renew your plan to continue using DentAI.",
        "sub_go_pricing": "💳 View Plans",
        "sub_active_until": "✅ Active until",
        "sub_free_trial": "🆓 Free trial",
        "sub_days_left": "days left",
        "admin_payments_menu": "💳 Payment Requests",
        "admin_pay_approve": "✅ Activate",
        "admin_pay_reject": "❌ Reject",
        "admin_pay_status": "Status",
        "admin_pay_plan": "Plan",
        "admin_pay_email": "Email",
        "admin_pay_date": "Date",
        "admin_pay_proof": "View Proof",
    },
    "FR": {
        "nav_home": "Accueil", "nav_contact": "Contact", "nav_login": "Connexion", "nav_signup": "Inscription",
        "sidebar_clinic": "🏥 Clinique DentAI", "sidebar_logout": "🚪 Déconnexion",
        "menu_dashboard": "📊 Tableau de bord", "menu_diagnosis": "🧠 Nouveau Diagnostic",
        "menu_historic": "📁 Historique", "menu_patients": "👨‍⚕️ Patients",
        "menu_profile": "👨‍⚕️ Profil ", "menu_settings": "⚙️ Paramètres",
        "logout_confirm": "Voulez-vous vraiment vous déconnecter ?",
        "logout_yes": "✅ Oui, déconnecter", "logout_cancel": "❌ Annuler",
        "dash_title": "📊 Vue d'ensemble", "dash_welcome": "Bienvenue Dr.",
        "dash_total_reports": "🧾 Rapports totaux", "dash_patients": "👤 Patients", "dash_today": "📅 Aujourd'hui",
        "dash_chart1": "🏆 Patients par nombre de rapports", "dash_chart2": "📈 Croissance des patients",
        "dash_no_patients": "Aucune donnée patient", "dash_no_growth": "Aucune donnée de croissance",
        "dash_recent": "🕒 Rapports récents", "dash_no_recent": "Aucun rapport récent",
        "dash_reports_per_patient": "Rapports par patient", "dash_num_reports": "Nombre de rapports",
        "dash_cumulative": "Croissance cumulative des patients", "dash_month": "Mois", "dash_total_p": "Total patients",
        "diag_header": "🧠 Diagnostic dentaire IA", "diag_upload_label": "Télécharger la radiographie",
        "diag_detect": "🔍 Détecter", "diag_analyse": "🧠 Analyser",
        "diag_detect_first": "⚠️ Veuillez d'abord exécuter **🔍 Détecter** avant d'analyser.",
        "diag_generating": "Génération du diagnostic IA...",
        "diag_num_detections": "Nombre de détections :",
        "diag_region": "Région", "diag_confidence": "Confiance",
        "diag_generated": "✅ Diagnostic IA généré",
        "diag_patient_info": "👤 Informations du patient",
        "diag_patient_name": "Nom du patient", "diag_patient_age": "Âge du patient",
        "diag_save_pdf": "📄 Enregistrer le rapport en PDF", "diag_fill_patient": "Veuillez remplir les infos patient",
        "diag_download": "⬇️ Télécharger le rapport", "diag_success": "Rapport généré avec succès !",
        "hist_header": "📁 Historique des rapports", "hist_no_reports": "Aucun rapport",
        "hist_patient": "Patient", "hist_date": "Date", "hist_xray": "Radiographie originale",
        "pat_header": "👨‍⚕️ Dossiers patients", "pat_search": "🔍 Rechercher un patient",
        "pat_no_found": "Aucun patient trouvé", "pat_no_reports": "Aucun rapport pour ce patient",
        "pat_view": "👁️ Voir", "pat_delete_all": "❌ Supprimer TOUS les rapports de",
        "pat_deleted": "Rapport supprimé", "pat_all_deleted": "Tous les rapports de",
        "pat_all_deleted2": "supprimés", "pat_download": "⬇️ Télécharger PDF",
        "pat_close": "❎ Fermer",
        "prof_header": "👨‍⚕️ Paramètres du profil", "prof_no_photo": "Aucune photo de profil",
        "prof_name": "Nom complet", "prof_email": "Email", "prof_dob": "Date de naissance",
        "prof_upload_photo": "Télécharger une photo", "prof_save": "💾 Mettre à jour le profil",
        "prof_updated": "✅ Profil mis à jour avec succès !", "prof_not_found": "Utilisateur introuvable.",
        "prof_go_login": "🚪 Aller à la connexion",
        "set_header": "⚙️ Paramètres avancés", "set_appearance": "🎨 Apparence",
        "set_dark_mode": "🌙 Mode sombre", "set_language": "🌐 Langue",
        "set_select_lang": "Choisir la langue", "set_notifications": "🔔 Notifications",
        "set_email_notif": "📧 Notifications par email", "set_recent_notif": "Notifications récentes :",
        "set_test_notif": "🧪 Ajouter une notification test", "set_security": "🔐 Sécurité",
        "set_current_pass": "Mot de passe actuel", "set_new_pass": "Nouveau mot de passe",
        "set_change_pass": "🔑 Changer le mot de passe", "set_pass_fill": "Veuillez remplir les deux champs",
        "set_pass_updated": "Mot de passe mis à jour !", "set_wrong_pass": "Mauvais mot de passe actuel",
        "set_2fa": "Activer 2FA (Démo)", "set_otp_label": "OTP de démo :",
        "set_otp_input": "Entrer l'OTP", "set_verify_otp": "Vérifier OTP",
        "set_otp_ok": "2FA vérifié !", "set_otp_fail": "OTP invalide",
        "set_save": "💾 Enregistrer les paramètres", "set_saved": "Paramètres enregistrés !",
        "set_not_logged": "Vous n'êtes pas connecté",
        # Pricing / Subscription
        "nav_pricing": "Tarifs",
        "pricing_title": "💎 Choisissez votre offre",
        "pricing_subtitle": "Commencez gratuitement, évoluez quand vous voulez.",
        "plan_free_title": "🆓 Essai gratuit",
        "plan_free_price": "0 DA",
        "plan_free_period": "3 jours",
        "plan_free_desc": "Essayez toutes les fonctionnalités pendant 3 jours.",
        "plan_monthly_title": "📅 Mensuel",
        "plan_monthly_price": "1 900 DA",
        "plan_monthly_period": "/ mois",
        "plan_monthly_desc": "Accès complet, facturé mensuellement.",
        "plan_yearly_title": "🏆 Annuel",
        "plan_yearly_price": "18 000 DA",
        "plan_yearly_period": "/ an",
        "plan_yearly_desc": "Meilleur rapport qualité-prix ! Économisez 3 600 DA.",
        "plan_btn_select": "Choisir",
        "plan_features": "✔ Diagnostic IA  ✔ Rapports PDF  ✔ Gestion patients  ✔ Tableau de bord",
        "plan_current": "✅ Plan actuel",
        "payment_title": "💳 Paiement – Virement manuel",
        "payment_info": "Veuillez effectuer le virement sur le compte ci-dessous puis remplir le formulaire.",
        "payment_bank": "🏦 Banque : CPA (Crédit Populaire d'Algérie)",
        "payment_rib": "🔢 RIB : 00799999000123456789 12",
        "payment_name": "👤 Titulaire : DentAI SAS",
        "payment_ref_label": "Référence de paiement (votre email)",
        "payment_amount_label": "Montant payé (DA)",
        "payment_proof_label": "Justificatif de paiement (capture d'écran / reçu)",
        "payment_submit": "📨 Envoyer",
        "payment_success": "✅ Paiement soumis ! Votre compte sera activé sous 24 h.",
        "payment_fill": "Veuillez remplir tous les champs et uploader le justificatif.",
        "sub_expired_title": "⏳ Votre abonnement a expiré",
        "sub_expired_msg": "Veuillez renouveler votre abonnement pour continuer.",
        "sub_go_pricing": "💳 Voir les offres",
        "sub_active_until": "✅ Actif jusqu'au",
        "sub_free_trial": "🆓 Essai gratuit",
        "sub_days_left": "jours restants",
        "admin_payments_menu": "💳 Demandes de paiement",
        "admin_pay_approve": "✅ Activer",
        "admin_pay_reject": "❌ Rejeter",
        "admin_pay_status": "Statut",
        "admin_pay_plan": "Offre",
        "admin_pay_email": "Email",
        "admin_pay_date": "Date",
        "admin_pay_proof": "Voir justificatif",
    },
    "AR": {
        "nav_home": "الرئيسية", "nav_contact": "اتصل بنا", "nav_login": "تسجيل الدخول", "nav_signup": "إنشاء حساب",
        "sidebar_clinic": "🏥 عيادة DentAI", "sidebar_logout": "🚪 تسجيل الخروج",
        "menu_dashboard": "📊 لوحة التحكم", "menu_diagnosis": "🧠 تشخيص جديد",
        "menu_historic": "📁 سجل التقارير", "menu_patients": "👨‍⚕️ المرضى",
        "menu_profile": "👨‍⚕️ الملف الشخصي ", "menu_settings": "⚙️ الإعدادات",
        "logout_confirm": "هل أنت متأكد من تسجيل الخروج؟",
        "logout_yes": "✅ نعم، خروج", "logout_cancel": "❌ إلغاء",
        "dash_title": "📊 نظرة عامة على العيادة", "dash_welcome": "مرحباً دكتور",
        "dash_total_reports": "🧾 إجمالي التقارير", "dash_patients": "👤 المرضى", "dash_today": "📅 اليوم",
        "dash_chart1": "🏆 المرضى حسب عدد التقارير", "dash_chart2": "📈 نمو المرضى",
        "dash_no_patients": "لا توجد بيانات مرضى", "dash_no_growth": "لا توجد بيانات نمو",
        "dash_recent": "🕒 التقارير الأخيرة", "dash_no_recent": "لا توجد تقارير حديثة",
        "dash_reports_per_patient": "التقارير لكل مريض", "dash_num_reports": "عدد التقارير",
        "dash_cumulative": "النمو التراكمي للمرضى", "dash_month": "الشهر", "dash_total_p": "إجمالي المرضى",
        "diag_header": "🧠 تشخيص أسنان بالذكاء الاصطناعي", "diag_upload_label": "رفع صورة الأشعة",
        "diag_detect": "🔍 كشف", "diag_analyse": "🧠 تحليل",
        "diag_detect_first": "⚠️ يرجى تشغيل **🔍 كشف** أولاً قبل التحليل.",
        "diag_generating": "جارٍ إنشاء التشخيص...",
        "diag_num_detections": "عدد الاكتشافات:",
        "diag_region": "المنطقة", "diag_confidence": "الثقة",
        "diag_generated": "✅ تم إنشاء التشخيص",
        "diag_patient_info": "👤 معلومات المريض",
        "diag_patient_name": "اسم المريض", "diag_patient_age": "عمر المريض",
        "diag_save_pdf": "📄 حفظ التقرير كـ PDF", "diag_fill_patient": "يرجى ملء معلومات المريض",
        "diag_download": "⬇️ تحميل التقرير", "diag_success": "تم إنشاء التقرير بنجاح!",
        "hist_header": "📁 سجل التقارير", "hist_no_reports": "لا توجد تقارير",
        "hist_patient": "المريض", "hist_date": "التاريخ", "hist_xray": "صورة الأشعة الأصلية",
        "pat_header": "👨‍⚕️ ملفات المرضى", "pat_search": "🔍 البحث عن مريض",
        "pat_no_found": "لم يُعثر على مرضى", "pat_no_reports": "لا توجد تقارير لهذا المريض",
        "pat_view": "👁️ عرض", "pat_delete_all": "❌ حذف جميع تقارير",
        "pat_deleted": "تم حذف التقرير", "pat_all_deleted": "تم حذف جميع تقارير",
        "pat_all_deleted2": "", "pat_download": "⬇️ تحميل PDF",
        "pat_close": "❎ إغلاق",
        "prof_header": "👨‍⚕️ إعدادات الملف الشخصي", "prof_no_photo": "لا توجد صورة شخصية",
        "prof_name": "الاسم الكامل", "prof_email": "البريد الإلكتروني", "prof_dob": "تاريخ الميلاد",
        "prof_upload_photo": "رفع صورة شخصية", "prof_save": "💾 تحديث الملف الشخصي",
        "prof_updated": "✅ تم تحديث الملف الشخصي بنجاح!", "prof_not_found": "المستخدم غير موجود.",
        "prof_go_login": "🚪 الذهاب لتسجيل الدخول",
        "set_header": "⚙️ الإعدادات المتقدمة", "set_appearance": "🎨 المظهر",
        "set_dark_mode": "🌙 الوضع الداكن", "set_language": "🌐 اللغة",
        "set_select_lang": "اختر اللغة", "set_notifications": "🔔 الإشعارات",
        "set_email_notif": "📧 إشعارات البريد الإلكتروني", "set_recent_notif": "الإشعارات الأخيرة:",
        "set_test_notif": "🧪 إضافة إشعار تجريبي", "set_security": "🔐 الأمان",
        "set_current_pass": "كلمة المرور الحالية", "set_new_pass": "كلمة المرور الجديدة",
        "set_change_pass": "🔑 تغيير كلمة المرور", "set_pass_fill": "يرجى ملء الحقلين",
        "set_pass_updated": "تم تحديث كلمة المرور!", "set_wrong_pass": "كلمة المرور الحالية خاطئة",
        "set_2fa": "تفعيل المصادقة الثنائية (تجريبي)", "set_otp_label": "رمز OTP التجريبي:",
        "set_otp_input": "أدخل رمز OTP", "set_verify_otp": "تحقق من OTP",
        "set_otp_ok": "تم التحقق!", "set_otp_fail": "رمز OTP غير صحيح",
        "set_save": "💾 حفظ الإعدادات", "set_saved": "تم حفظ الإعدادات!",
        "set_not_logged": "أنت لست مسجلاً للدخول",
        # Pricing / Subscription
        "nav_pricing": "الأسعار",
        "pricing_title": "💎 اختر خطتك",
        "pricing_subtitle": "ابدأ مجاناً، وقم بالترقية في أي وقت.",
        "plan_free_title": "🆓 تجربة مجانية",
        "plan_free_price": "0 دج",
        "plan_free_period": "3 أيام",
        "plan_free_desc": "جرّب جميع الميزات مجاناً لمدة 3 أيام بعد الموافقة.",
        "plan_monthly_title": "📅 شهري",
        "plan_monthly_price": "1 900 دج",
        "plan_monthly_period": "/ شهر",
        "plan_monthly_desc": "وصول كامل لجميع الميزات، يُفوتر شهرياً.",
        "plan_yearly_title": "🏆 سنوي",
        "plan_yearly_price": "18 000 دج",
        "plan_yearly_period": "/ سنة",
        "plan_yearly_desc": "أفضل قيمة! وفّر 3 600 دج مقارنة بالاشتراك الشهري.",
        "plan_btn_select": "اختر الخطة",
        "plan_features": "✔ تشخيص ذكاء اصطناعي  ✔ تقارير PDF  ✔ إدارة المرضى  ✔ لوحة التحكم",
        "plan_current": "✅ الخطة الحالية",
        "payment_title": "💳 الدفع – تحويل يدوي",
        "payment_info": "يرجى تحويل المبلغ إلى الحساب التالي ثم ملء النموذج أدناه.",
        "payment_bank": "🏦 البنك: القرض الشعبي الجزائري (CPA)",
        "payment_rib": "🔢 رقم الحساب: 00799999000123456789 12",
        "payment_name": "👤 اسم صاحب الحساب: DentAI SAS",
        "payment_ref_label": "مرجع الدفع (بريدك الإلكتروني)",
        "payment_amount_label": "المبلغ المدفوع (دج)",
        "payment_proof_label": "رفع إثبات الدفع (لقطة شاشة / وصل)",
        "payment_submit": "📨 إرسال الطلب",
        "payment_success": "✅ تم إرسال طلب الدفع! سيتم تفعيل حسابك خلال 24 ساعة.",
        "payment_fill": "يرجى ملء جميع الحقول ورفع إثبات الدفع.",
        "sub_expired_title": "⏳ انتهى اشتراكك",
        "sub_expired_msg": "يرجى تجديد اشتراكك لمواصلة استخدام DentAI.",
        "sub_go_pricing": "💳 عرض الخطط",
        "sub_active_until": "✅ نشط حتى",
        "sub_free_trial": "🆓 تجربة مجانية",
        "sub_days_left": "أيام متبقية",
        "admin_payments_menu": "💳 طلبات الدفع",
        "admin_pay_approve": "✅ تفعيل",
        "admin_pay_reject": "❌ رفض",
        "admin_pay_status": "الحالة",
        "admin_pay_plan": "الخطة",
        "admin_pay_email": "البريد الإلكتروني",
        "admin_pay_date": "التاريخ",
        "admin_pay_proof": "عرض الإثبات",
    },
}

def t(key):
    """Return translated string for current session language."""
    lang = st.session_state.get("active_lang", "EN")
    return TRANSLATIONS.get(lang, TRANSLATIONS["EN"]).get(key, key)

def apply_rtl():
    """Inject RTL CSS for Arabic."""
    st.markdown("""
    <style>
    .stApp, .stApp * { direction: rtl; text-align: right; }
    .stSidebar, .stSidebar * { direction: rtl; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# =======================
# 🗄️ DATABASE HELPERS
# =======================

@contextlib.contextmanager
def get_db():
    """Context manager — always closes the connection even on exceptions."""
    conn = sqlite3.connect("dentai.db", timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_query(query, params=()):
    """Execute an INSERT / UPDATE / DELETE / CREATE statement."""
    with get_db() as conn:
        conn.execute(query, params)


def run_read(query, params=()):
    """Execute a SELECT and return a DataFrame."""
    with get_db() as conn:
        return pd.read_sql_query(query, conn, params=params)


# =======================
# 🔑 model functions
# =======================
def create_boxed_copies(original_img, boxes):
    boxed_images = []

    for box in boxes:
        img_copy = original_img.copy()

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label = f"{model.names[cls_id]} {conf:.2f}"

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        boxed_images.append(img_copy)

    return boxed_images


def get_ai_diagnosis(detected_classes, boxed_images):
    content = []

    prompt = f"""
You are a professional dental specialist.

You are given multiple dental X-ray images.
Each image highlights one detected region.

Detected conditions:
{', '.join(detected_classes)}

For each region give:

1. confidance:
2. professional diagnosis 
3. Global risk explanation
4. Integrated treatment plan

Rules:
- Analyze each region separately
- Maximum 255 words
- Medical professional tone
- Structured response

"""

    content.append({"type": "text", "text": prompt})

    for img in boxed_images:
        _, buffer = cv2.imencode(".jpg", img)
        base64_img = base64.b64encode(buffer).decode()

        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}]
    )

    return response.choices[0].message.content


# =======================
# 🔑 pdf function
# =======================
def generate_pdf_report(doctor_name, patient_name, patient_age, diagnosis_text, annotated_image_path):
    os.makedirs("reports_pdf", exist_ok=True)
    
    now = datetime.now()
    file_name = f"reports_pdf/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    date_report=now.strftime("%B %D, %Y")

    doc = SimpleDocTemplate(
        file_name,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    content = []

    if os.path.exists("C:/Users/computer house 41/Desktop/logo/0.png"):
        logo = RLImage("C:/Users/computer house 41/Desktop/logo/0.png", width=90, height=90)
        content.append(logo)

    content.append(Spacer(1, 20))
    content.append(build_header(doctor_name, patient_name, patient_age, date_report))
    content.append(Spacer(1, 10))
    

    content.append(Paragraph("<b>Detected X-ray</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    if annotated_image_path and os.path.exists(annotated_image_path):
        content.append(RLImage(annotated_image_path, width=450, height=320))
    else:
        content.append(Paragraph("No detected image available", styles["Normal"]))

    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Diagnosis</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for line in diagnosis_text.split("\n"):
        if line.strip():
            content.append(Paragraph(line, styles["Normal"]))
            content.append(Spacer(1, 5))

    doc.build(
        content,
        onFirstPage=add_page_decorations,
        onLaterPages=add_page_decorations
    )

    return file_name


# =======================
# 🔑 RESEND API
# =======================
resend.api_key = "re_WPTpx2Xa_7dGvB4sCoDHo1cSt5jiZX4wh"


def send_email(user_email, message):
    try:
        resend.Emails.send({
            "from": "DentAI <onboarding@resend.dev>",
            "to": ["amelladjailia58@gmail.com"],
            "subject": "New Contact Message",
            "html": f"<p><b>From:</b> {user_email}</p><p>{message}</p>"
        })
        return True
    except Exception as e:
        return str(e)


# =======================
# 👨‍⚕️ show pdf
# =======================
@st.dialog("📄 Report Viewer")
def show_pdf(pdf_path, report_id):
    import streamlit.components.v1 as components

    if pdf_path and os.path.exists(pdf_path):

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        if len(pdf_bytes) == 0:
            st.error("PDF file is empty")
            return

        base64_pdf = base64.b64encode(pdf_bytes).decode()

        pdf_html = f"""
        <html>
        <body style="margin:0">
            <embed
                src="data:application/pdf;base64,{base64_pdf}"
                width="100%"
                height="700px"
                type="application/pdf">
            </embed>
        </body>
        </html>
        """

        components.html(pdf_html, height=750)

        st.download_button(
            "⬇️ Download PDF",
            pdf_bytes,
            file_name=os.path.basename(pdf_path),
            key=f"download_{report_id}"
        )

    else:
        st.error("PDF not found or path is incorrect")


# =======================
# 🗄️ DATABASE SETUP
# =======================

run_query("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT,
    phone TEXT,
    email TEXT UNIQUE,
    password TEXT,
    diploma TEXT,
    status TEXT
)
""")

try:
    run_query("ALTER TABLE users ADD COLUMN dob TEXT")
except Exception:
    pass

try:
    run_query("ALTER TABLE users ADD COLUMN profile_image TEXT")
except Exception:
    pass

run_query("""
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    patient_name TEXT,
    image_path TEXT,
    diagnosis TEXT,
    confidence TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

try:
    run_query("ALTER TABLE reports ADD COLUMN pdf_path TEXT")
except Exception:
    pass

try:
    run_query("ALTER TABLE reports ADD COLUMN original_image_path TEXT")
except Exception:
    pass

run_query("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age TEXT,
    notes TEXT
)
""")

run_query("""
CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE,
    dark_mode INTEGER DEFAULT 0,
    email_notifications INTEGER DEFAULT 1
)
""")

try:
    run_query("ALTER TABLE settings ADD COLUMN language TEXT DEFAULT 'EN'")
except Exception:
    pass

try:
    run_query("ALTER TABLE settings ADD COLUMN notifications TEXT DEFAULT ''")
except Exception:
    pass

try:
    run_query("ALTER TABLE settings ADD COLUMN two_factor INTEGER DEFAULT 0")
except Exception:
    pass

# =======================
# 💳 Subscription tables
# =======================
run_query("""
CREATE TABLE IF NOT EXISTS subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE,
    plan TEXT DEFAULT 'free_trial',
    start_date TEXT,
    end_date TEXT,
    status TEXT DEFAULT 'active'
)
""")

run_query("""
CREATE TABLE IF NOT EXISTS payment_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    plan TEXT,
    amount INTEGER,
    reference TEXT,
    proof_path TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# =======================
# 💳 Subscription helpers
# =======================
from datetime import timedelta

def get_subscription(user_id):
    """Return (plan, end_date, status, days_left) or None if no subscription."""
    data = run_read("SELECT * FROM subscriptions WHERE user_id=?", (user_id,))
    if data.empty:
        return None
    row = data.iloc[0]
    end_date = datetime.strptime(row["end_date"], "%Y-%m-%d").date()
    today = datetime.today().date()
    days_left = (end_date - today).days
    status = "active" if days_left >= 0 else "expired"
    return {
        "plan": row["plan"],
        "end_date": end_date,
        "status": status,
        "days_left": days_left
    }

def create_free_trial(user_id):
    """Give 3-day free trial starting today."""
    start = datetime.today().date()
    end   = start + timedelta(days=3)
    try:
        run_query("""
            INSERT OR IGNORE INTO subscriptions (user_id, plan, start_date, end_date, status)
            VALUES (?, 'free_trial', ?, ?, 'active')
        """, (user_id, str(start), str(end)))
    except Exception:
        pass

def activate_subscription(user_id, plan):
    """Activate monthly or yearly subscription from today."""
    start = datetime.today().date()
    if plan == "monthly":
        end = start + timedelta(days=30)
    else:  # yearly
        end = start + timedelta(days=365)
    run_query("""
        INSERT INTO subscriptions (user_id, plan, start_date, end_date, status)
        VALUES (?, ?, ?, ?, 'active')
        ON CONFLICT(user_id) DO UPDATE SET plan=excluded.plan,
            start_date=excluded.start_date, end_date=excluded.end_date, status='active'
    """, (user_id, plan, str(start), str(end)))

def check_subscription_access(user_id):
    """Returns True if user has active subscription, False otherwise."""
    sub = get_subscription(user_id)
    if sub is None:
        create_free_trial(user_id)
        return True
    return sub["status"] == "active"

# =======================
# 📩 Contact messages table & helpers
# =======================
run_query("""
CREATE TABLE IF NOT EXISTS contact_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    sender_name TEXT,
    sender_email TEXT,
    message TEXT,
    reply TEXT DEFAULT '',
    is_read INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

def save_contact_message(user_id, sender_name, sender_email, message):
    run_query("""
        INSERT INTO contact_messages (user_id, sender_name, sender_email, message)
        VALUES (?, ?, ?, ?)
    """, (user_id, sender_name, sender_email, message))

def get_unread_count():
    df = run_read("SELECT COUNT(*) as cnt FROM contact_messages WHERE is_read=0")
    return int(df.iloc[0]["cnt"]) if not df.empty else 0

# Create admin account
admin_email = "admin@dentai.com"
admin_password = hashlib.sha256("admin".encode()).hexdigest()

try:
    run_query("""
        INSERT INTO users (full_name, phone, email, password, diploma, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("Admin", "0000000000", admin_email, admin_password, "admin.png", "approved"))
    print("Admin created ✅")
except Exception:
    print("Admin already exists")


# =======================
# 💾 PDF page decorations
# =======================
from reportlab.pdfgen import canvas


def add_page_decorations(canvas, doc):
    canvas.saveState()

    canvas.setFont("Helvetica", 60)
    canvas.setFillGray(0.9, 0.5)
    canvas.drawCentredString(300, 400, "DentAI Clinic")

    page_num = canvas.getPageNumber()
    canvas.setFont("Helvetica", 9)
    canvas.setFillGray(0)
    canvas.drawRightString(580, 20, f"Page {page_num}")

    canvas.restoreState()


# =======================
# 💾 PDF header
# =======================
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors


def build_header(doctor_name, patient_name, patient_age, date_report):
    data = [
        [
            f"Doctor: {doctor_name}",
            f"Patient: {patient_name} | Age: {patient_age}",
            f"date:{date_report}"
        ]
    ]

    table = Table(data, colWidths=[160, 240, 135])

    table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("ALIGN", (1, 0), (1, 0), "CENTER"),
        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    return table


# =======================
# 💾 Settings helpers
# =======================
def get_settings(user_id):
    data = run_read("""
        SELECT dark_mode, email_notifications, language, notifications, two_factor
        FROM settings WHERE user_id=?
    """, (user_id,))

    if data.empty:
        run_query("""
            INSERT INTO settings (user_id, dark_mode, email_notifications, language, notifications, two_factor)
            VALUES (?, 0, 1, 'EN', '', 0)
        """, (user_id,))
        return 0, 1, "EN", "", 0

    row = data.iloc[0]
    return (
        row["dark_mode"],
        row["email_notifications"],
        row["language"],
        row["notifications"],
        row["two_factor"]
    )


def update_settings(user_id, dark_mode, email_notifications, language, notifications, two_factor):
    run_query("""
        UPDATE settings
        SET dark_mode=?, email_notifications=?, language=?, notifications=?, two_factor=?
        WHERE user_id=?
    """, (dark_mode, email_notifications, language, notifications, two_factor, user_id))


# =======================
# 💾 User helpers
# =======================
def update_user(user_id, name, email, dob, image_file=None):
    image_path = None

    if image_file:
        os.makedirs("profiles", exist_ok=True)
        image_path = f"profiles/{user_id}_profile.png"

        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())

        run_query("""
            UPDATE users
            SET full_name=?, email=?, dob=?, profile_image=?
            WHERE id=?
        """, (name, email, dob, image_path, user_id))

    else:
        run_query("""
            UPDATE users
            SET full_name=?, email=?, dob=?
            WHERE id=?
        """, (name, email, dob, user_id))

    return True


# =======================
# 💾 Report helpers
# =======================
def save_report(user_id, patient_name, image_path, diagnosis, pdf_path, confidence="", original_image_path=""):
    try:
        run_query("""
            INSERT INTO reports (user_id, patient_name, image_path, diagnosis, confidence, pdf_path, original_image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, patient_name, image_path, diagnosis, confidence, pdf_path, original_image_path))
        return True
    except Exception as e:
        return str(e)

############

# =======================
# 💾 Sign-up helper
# =======================
def save_user(full_name, phone, email, password, diploma_file):
    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{email}_diploma.png"

    with open(file_path, "wb") as f:
        f.write(diploma_file.getbuffer())

    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    try:
        run_query("""
            INSERT INTO users (full_name, phone, email, password, diploma, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (full_name, phone, email, hashed_password, file_path, "pending"))
        return True
    except Exception as e:
        return str(e)


# =======================
# ⚙️ PAGE CONFIG
# =======================
st.set_page_config(page_title="DentAI", layout="wide")

# =======================
# 🔁 SESSION
# =======================
if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page):
    st.session_state.page = page
    

# =======================
# 🖼️ BACKGROUND
# =======================
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


bg = get_base64_image("logo/10.png")

# =======================
# 🎨 CSS
# =======================
# ============================================================
# Replace your existing CSS  st.markdown(f"""<style>...""")
# block with this one  (the one that starts with .stApp)
# ============================================================

st.markdown(f"""
<style>

/* ── Background ─────────────────────────────────────── */
.stApp {{
    background-image: url("data:image/png;base64,{bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* ══════════════════════════════════════════════════════
   BUTTON BASE  — every stButton
   ══════════════════════════════════════════════════════ */
div.stButton > button {{
    background: linear-gradient(135deg, #f0f4ff 0%, #e8eef8 100%);
    border: 2px solid #7aa3d4;
    color: #1a3a5c;
    font-weight: 600;
    font-size: 14px;
    border-radius: 10px;
    padding: 8px 20px;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(44, 123, 229, 0.15);
}}

div.stButton > button:hover {{
    background: linear-gradient(135deg, #2c7be5 0%, #5b9bd5 100%);
    border-color: #1a5edb;
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(44, 123, 229, 0.40);
    transform: translateY(-1px);
}}

div.stButton > button:active {{
    transform: translateY(0px);
    box-shadow: 0 2px 6px rgba(44, 123, 229, 0.25);
}}

/* ══════════════════════════════════════════════════════
   SIDEBAR buttons  — slightly darker frame
   ══════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] div.stButton > button {{
    background: linear-gradient(135deg, #e4ecf7 0%, #d6e4f0 100%);
    border: 2px solid #5b8fbf;
    color: #1a3a5c;
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
}}

section[data-testid="stSidebar"] div.stButton > button:hover {{
    background: linear-gradient(135deg, #1a5edb 0%, #2c7be5 100%);
    border-color: #1a5edb;
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(26, 94, 219, 0.35);
}}

/* ══════════════════════════════════════════════════════
   LOGOUT / DANGER buttons  — red-tinted frame
   Match any button whose text contains "Logout" / "Delete"
   via the title attribute Streamlit adds
   ══════════════════════════════════════════════════════ */
div.stButton > button[title*="Logout"],
div.stButton > button[title*="Delete"],
div.stButton > button[title*="Déconnexion"],
div.stButton > button[title*="Supprimer"] {{
    border-color: #c0392b;
    color: #c0392b;
}}

div.stButton > button[title*="Logout"]:hover,
div.stButton > button[title*="Delete"]:hover {{
    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    border-color: #c0392b;
    color: #ffffff;
}}

/* ══════════════════════════════════════════════════════
   SUCCESS / APPROVE buttons  — green-tinted frame
   ══════════════════════════════════════════════════════ */
div.stButton > button[title*="Approve"],
div.stButton > button[title*="Activer"],
div.stButton > button[title*="Yes"] {{
    border-color: #27ae60;
    color: #1a7a40;
}}

div.stButton > button[title*="Approve"]:hover,
div.stButton > button[title*="Activer"]:hover {{
    background: linear-gradient(135deg, #27ae60 0%, #1e8449 100%);
    border-color: #1e8449;
    color: #ffffff;
}}

/* ══════════════════════════════════════════════════════
   PRIMARY CTA  — "Start Analyzing", "Login", "Send"
   ══════════════════════════════════════════════════════ */
.start-btn div.stButton > button {{
    background: linear-gradient(135deg, #2c7be5 0%, #1a5edb 100%) !important;
    border: 2px solid #1a5edb !important;
    color: white !important;
    padding: 14px 30px !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    box-shadow: 0 4px 16px rgba(44, 123, 229, 0.45) !important;
}}

.start-btn div.stButton > button:hover {{
    background: linear-gradient(135deg, #1a5edb 0%, #1348b0 100%) !important;
    box-shadow: 0 6px 20px rgba(44, 123, 229, 0.60) !important;
    transform: translateY(-2px);
}}

/* ══════════════════════════════════════════════════════
   DOWNLOAD buttons
   ══════════════════════════════════════════════════════ */
div.stDownloadButton > button {{
    background: linear-gradient(135deg, #eaf4ea 0%, #d5ecd5 100%);
    border: 2px solid #27ae60;
    color: #1a7a40;
    font-weight: 600;
    border-radius: 10px;
    padding: 8px 20px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(39, 174, 96, 0.15);
}}

div.stDownloadButton > button:hover {{
    background: linear-gradient(135deg, #27ae60 0%, #1e8449 100%);
    border-color: #1e8449;
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(39, 174, 96, 0.40);
    transform: translateY(-1px);
}}

/* ══════════════════════════════════════════════════════
   MISC helpers
   ══════════════════════════════════════════════════════ */
.center {{
    text-align: center;
    margin-top: 100px;
}}

.overlay {{
    background-color: rgba(255,255,255,0.75);
    padding: 40px;
    border-radius: 15px;
    width: 60%;
    margin: auto;
}}

</style>
""", unsafe_allow_html=True)

# =======================
# 🧭 NAVBAR
# =======================
col1, col2 = st.columns([2, 5])

with col1:
    c1, c2 = st.columns([1, 2])

    with c1:
        try:
            st.image("logo/0.png", width=60)
        except Exception:
            st.write("🦷")

    with c2:
        st.markdown("### DentAI")

# ============================================================
# Replace your entire navbar  col2  block with this
# ============================================================

with col2:
    nav1, nav2, nav3, nav4, nav5 = st.columns(5)

    with nav1:
        st.markdown('<div class="navbar-btn">', unsafe_allow_html=True)
        if st.button("Home", key="home_btn"):
            go_to("home")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav2:
        st.markdown('<div class="navbar-btn">', unsafe_allow_html=True)
        if st.button("Pricing", key="pricing_btn"):
            go_to("pricing")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav3:
        st.markdown('<div class="navbar-btn">', unsafe_allow_html=True)
        if st.button("Contact", key="contact_btn"):
            go_to("contact")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav4:
        st.markdown('<div class="navbar-btn">', unsafe_allow_html=True)
        if st.button("Login", key="nav_login_btn"):
            go_to("login")
        st.markdown('</div>', unsafe_allow_html=True)

    with nav5:
        st.markdown('<div class="navbar-btn">', unsafe_allow_html=True)
        if st.button("Sign Up", key="nav_signup_btn"):
            go_to("signup")
        st.markdown('</div>', unsafe_allow_html=True)

# =======================
# 🏠 HOME
# =======================
if st.session_state.page == "home":

    st.markdown("""
    ## 🧠 About DentAI

    DentAI is an intelligent platform designed to assist dental professionals  
    in analyzing dental X-ray images using advanced artificial intelligence.

    It combines object detection using YOLO and AI-powered diagnosis  
    to provide fast, accurate, and reliable results.

    🔹 Detects dental issues automatically  
    🔹 Provides AI-based diagnosis  
    🔹 Generates professional medical reports  
    """)

    st.markdown('<div class="start-btn">', unsafe_allow_html=True)
    if st.button("🚀 Start Analyzing", key="home_start_btn"):
        go_to("login")
    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# 📞 CONTACT
# =======================

elif st.session_state.page == "contact":

    st.title("📞 Contact Us")
    st.write("Send us your message and we will receive it instantly.")

    logged_in      = "user" in st.session_state
    default_email  = st.session_state.user["email"] if logged_in else ""
    default_name   = st.session_state.user["name"]  if logged_in else ""
    sender_user_id = st.session_state.user["id"]    if logged_in else None

    # ── BACK button (only for logged-in users) ───────────────
    if logged_in:
        if st.button("⬅️ Back to Dashboard", key="contact_back_btn"):
            go_to("dashboard")
            st.rerun()

    user_name  = st.text_input("Your Name",  value=default_name,  key="contact_name")
    user_email = st.text_input("Your Email", value=default_email, key="contact_email",
                               disabled=logged_in)
    message    = st.text_area("Your Message", key="contact_msg")

    if st.button("Send Message", key="contact_send_btn"):
        if user_email and message and user_name:
            save_contact_message(sender_user_id, user_name, user_email, message)
            with st.spinner("Sending..."):
                result = send_email(user_email, message)
            st.success("✅ Message sent successfully! We'll get back to you soon.")
        else:
            st.warning("Please fill all fields")

    # ── Show previous messages + admin replies (logged-in only) ──
    if logged_in and sender_user_id:
        st.markdown("---")
        st.subheader("📬 Your Previous Messages")

        my_msgs = run_read("""
            SELECT sender_name, message, reply, created_at
            FROM contact_messages
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (sender_user_id,))

        if my_msgs.empty:
            st.info("No messages yet.")
        else:
            for _, row in my_msgs.iterrows():
                with st.expander(f"📨 {row['created_at']}"):
                    st.markdown(f"**Your message:**")
                    st.info(row["message"])

                    reply = str(row["reply"]).strip() if row["reply"] else ""
                    if reply:
                        st.markdown("**↩️ Admin reply:**")
                        st.success(reply)

                        # Also send reply to user's real email the first time
                        # (admin already triggers this from admin panel,
                        #  so no extra send needed here — just display)
                    else:
                        st.caption("⏳ No reply yet.")

# =======================
# 🔐 LOGIN
# =======================
elif st.session_state.page == "login":

    st.title("🔐 Login")

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login", key="login_btn"):

        if email and password:

            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            user_df = run_read("""
                SELECT * FROM users
                WHERE email=? AND password=?
            """, (email, hashed_password))

            if not user_df.empty:

                user = user_df.iloc[0]

                if email == "admin@dentai.com":
                    st.success("Welcome Admin 👑")
                    st.session_state.admin = True
                    go_to("admin")
                    st.rerun()

                if user["status"] != "approved":
                    st.warning("⏳ Your account is waiting for admin approval")
                    st.stop()

                st.success("Login successful!")

                st.session_state.user = {
                    "id":    int(user["id"]),
                    "name":  user["full_name"],
                    "email": user["email"]
                }

                # Create free trial if no subscription exists
                create_free_trial(int(user["id"]))

                go_to("dashboard")
                st.rerun()

            else:
                st.error("❌ Invalid email or password")

        else:
            st.warning("⚠️ Fill all fields")

# =======================
# 📝 SIGNUP
# =======================
elif st.session_state.page == "signup":

    st.title("📝 Dentist Registration")

    full_name = st.text_input("Full Name", key="su_name")
    phone = st.text_input("Phone", key="su_phone")
    email = st.text_input("Email", key="su_email")
    password = st.text_input("Password", type="password", key="su_pass")
    diploma = st.file_uploader("Diploma", type=["png", "jpg", "jpeg"], key="su_diploma")

    if st.button("Create Account", key="signup_btn"):

        if full_name and phone and email and password and diploma:

            existing = run_read(
                "SELECT id FROM users WHERE email=?",
                (email,)
            )

            if not existing.empty:
                st.error("❌ Email already exists. Please use another one.")
            else:
                result = save_user(full_name, phone, email, password, diploma)

                if result is True:
                    st.success("Account created! Waiting for approval.")
                else:
                    st.error(f"Error: {result}")

        else:
            st.warning("Fill all fields")

# =======================
# 📊 ADMIN PANEL
# =======================
elif st.session_state.page == "admin":

    st.title("🛠️ Admin Panel")

    if "admin" not in st.session_state:
        st.error("Access denied")
        st.stop()

    with st.sidebar:

        st.markdown("## 👑 Admin Control Panel")

        # Unread messages badge
        _unread = get_unread_count()
        _msg_label = f"📩 Messages {'🔴 ' + str(_unread) if _unread > 0 else ''}"

        admin_menu = st.radio(
            "Navigation",
            [
                "📊 Dashboard",
                "👥 All Users",
                "⏳ Pending Accounts",
                "💳 Payment Requests",
                _msg_label,
            ]
        )

        st.markdown("---")

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            go_to("home")
            st.rerun()

    if admin_menu == "📊 Dashboard":

        st.info("Welcome to DentAI Admin Dashboard 👑")
        st.subheader("📊 System Overview")

        users_df   = run_read("SELECT * FROM users")
        reports_df = run_read("SELECT * FROM reports")
        pay_df     = run_read("SELECT * FROM payment_requests")

        total_users = len(users_df)
        pending  = len(users_df[users_df["status"] == "pending"])
        approved = len(users_df[users_df["status"] == "approved"])
        rejected = len(users_df[users_df["status"] == "rejected"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Users",    total_users)
        c2.metric("⏳ Pending",  pending)
        c3.metric("✅ Approved", approved)
        c4.metric("❌ Rejected", rejected)

        st.divider()

        # ── Payment Overview ──────────────────────────────────────────────
        st.subheader("💳 Payment Overview")

        if pay_df.empty:
            total_revenue   = 0
            pay_pending_n   = 0
            pay_approved_n  = 0
            pay_rejected_n  = 0
        else:
            approved_pay   = pay_df[pay_df["status"] == "approved"]
            total_revenue  = int(approved_pay["amount"].sum()) if not approved_pay.empty else 0
            pay_pending_n  = len(pay_df[pay_df["status"] == "pending"])
            pay_approved_n = len(approved_pay)
            pay_rejected_n = len(pay_df[pay_df["status"] == "rejected"])

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("💰 Total Revenue",     f"{total_revenue:,} DA")
        p2.metric("⏳ Pending Payments",  pay_pending_n)
        p3.metric("✅ Approved Payments", pay_approved_n)
        p4.metric("❌ Rejected Payments", pay_rejected_n)

        st.divider()

        # ── Subscriptions Overview ────────────────────────────────────────
        st.subheader("📋 Active Subscriptions")

        subs_df = run_read("""
            SELECT u.full_name, s.plan, s.start_date, s.end_date, s.status
            FROM subscriptions s
            JOIN users u ON s.user_id = u.id
            ORDER BY s.end_date DESC
        """)

        if subs_df.empty:
            st.info("No subscriptions yet.")
        else:
            total_active  = len(subs_df[subs_df["status"] == "active"])
            monthly_subs  = len(subs_df[subs_df["plan"] == "monthly"])
            yearly_subs   = len(subs_df[subs_df["plan"] == "yearly"])
            trial_subs    = len(subs_df[subs_df["plan"] == "free_trial"])

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("✅ Active Subs",   total_active)
            s2.metric("🆓 Free Trials",   trial_subs)
            s3.metric("📅 Monthly",       monthly_subs)
            s4.metric("🏆 Yearly",        yearly_subs)

            subs_df.columns = ["Doctor", "Plan", "Start", "End", "Status"]
            st.dataframe(subs_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── 3 charts in one row ───────────────────────────────────────────
        ch1, ch2, ch3 = st.columns(3)

        # Chart 1 – Payment status pie
        with ch1:
            st.markdown("**Payment Status Distribution**")
            if not pay_df.empty:
                status_counts = pay_df["status"].value_counts()
                colors_map   = {"approved": "#28a745", "pending": "#ffc107", "rejected": "#dc3545"}
                wedge_colors = [colors_map.get(s, "#999") for s in status_counts.index]
                fig_pay, ax_pay = plt.subplots(figsize=(3.5, 3.5))
                ax_pay.pie(
                    status_counts.values,
                    labels=status_counts.index,
                    autopct="%1.1f%%",
                    colors=wedge_colors,
                    startangle=90
                )
                ax_pay.set_ylabel("")
                fig_pay.tight_layout()
                st.pyplot(fig_pay)
            else:
                st.info("No payment data yet.")

        # Chart 2 – Revenue by plan bar
        with ch2:
            st.markdown("**Revenue by Plan (Approved)**")
            if not pay_df.empty:
                rev_by_plan = (
                    pay_df[pay_df["status"] == "approved"]
                    .groupby("plan")["amount"]
                    .sum()
                    .reset_index()
                )
                if rev_by_plan.empty:
                    st.info("No approved payments yet.")
                else:
                    fig_rev, ax_rev = plt.subplots(figsize=(3.5, 3.5))
                    bar_colors = ["#2c7be5" if p == "monthly" else "#28a745" for p in rev_by_plan["plan"]]
                    bars = ax_rev.bar(rev_by_plan["plan"], rev_by_plan["amount"], color=bar_colors)
                    ax_rev.bar_label(bars, labels=[f"{v:,} DA" for v in rev_by_plan["amount"]], padding=4, fontsize=9)
                    ax_rev.set_ylabel("Amount (DA)")
                    ax_rev.set_title("Revenue by Plan")
                    fig_rev.tight_layout()
                    st.pyplot(fig_rev)
            else:
                st.info("No payment data yet.")

        # Chart 3 – Users status pie
        with ch3:
            st.markdown("**Users Status Distribution**")
            fig_usr, ax_usr = plt.subplots(figsize=(3.5, 3.5))
            users_df["status"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax_usr)
            ax_usr.set_ylabel("")
            fig_usr.tight_layout()
            st.pyplot(fig_usr)

        st.divider()

    if admin_menu == "👥 All Users":

        st.subheader("👥 Manage Users")

        users_df = run_read("SELECT * FROM users")

        col1, col2 = st.columns(2)
        search        = col1.text_input("🔍 Search user")
        status_filter = col2.selectbox("Filter by status", ["all", "pending", "approved", "rejected"])

        if search:
            users_df = users_df[users_df["full_name"].str.contains(search, case=False)]

        if status_filter != "all":
            users_df = users_df[users_df["status"] == status_filter]

        for _, user in users_df.iterrows():

            with st.expander(f"{user['full_name']} ({user['status']})"):

                st.write("📧", user["email"])
                st.write("📞", user["phone"])
                st.write("📌 Status:", user["status"])

                if st.button("🗑️ Delete User", key=f"del_{user['id']}"):
                    run_query("DELETE FROM users WHERE id=?",        (int(user["id"]),))
                    run_query("DELETE FROM reports WHERE user_id=?", (int(user["id"]),))
                    st.warning("User deleted ❌")
                    st.rerun()

    if admin_menu == "⏳ Pending Accounts":

        st.subheader("⏳ Pending Approvals")

        users_df = run_read("SELECT * FROM users WHERE status='pending'")

        if users_df.empty:
            st.success("No pending accounts 🎉")

        else:
            for _, user in users_df.iterrows():

                with st.container():

                    st.write(f"👤 {user['full_name']}")
                    st.write(f"📧 {user['email']}")

                    if user["diploma"] and os.path.exists(str(user["diploma"])):
                        st.image(user["diploma"], width=200)

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("✅ Approve", key=f"ap_{user['id']}"):
                            run_query("UPDATE users SET status='approved' WHERE id=?", (int(user["id"]),))
                            st.success("Approved")
                            st.rerun()

                    with col2:
                        if st.button("❌ Reject", key=f"re_{user['id']}"):
                            run_query("UPDATE users SET status='rejected' WHERE id=?", (int(user["id"]),))
                            st.error("Rejected")
                            st.rerun()

                st.divider()

    if admin_menu == "💳 Payment Requests":

        st.subheader("💳 Payment Requests")

        pay_df = run_read("""
            SELECT pr.id, u.full_name, u.email, pr.plan, pr.amount, pr.reference,
                   pr.proof_path, pr.status, pr.created_at, pr.user_id
            FROM payment_requests pr
            JOIN users u ON pr.user_id = u.id
            ORDER BY pr.created_at DESC
        """)

        if pay_df.empty:
            st.info("No payment requests yet.")
        else:
            filter_status = st.selectbox("Filter", ["all", "pending", "approved", "rejected"], key="pay_filter")
            if filter_status != "all":
                pay_df = pay_df[pay_df["status"] == filter_status]

            for _, row in pay_df.iterrows():
                badge = "✅" if row["status"] == "approved" else ("⏳" if row["status"] == "pending" else "❌")
                with st.expander(f"{badge} {row['full_name']} — {row['plan'].upper()} — {row['amount']} DA"):
                    st.write(f"📧 {row['email']}")
                    st.write(f"🗓️ {row['created_at']}")
                    st.write(f"🔖 Ref: {row['reference']}")
                    st.write(f"📌 Status: **{row['status']}**")

                    if row["proof_path"] and os.path.exists(str(row["proof_path"])):
                        if str(row["proof_path"]).lower().endswith(".pdf"):
                            st.write("📄 PDF proof uploaded")
                        else:
                            st.image(row["proof_path"], width=300, caption="Payment Proof")

                    if row["status"] == "pending":
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("✅ Approve", key=f"pay_ap_{row['id']}"):
                                run_query(
                                    "UPDATE payment_requests SET status='approved' WHERE id=?",
                                    (int(row["id"]),)
                                )
                                activate_subscription(int(row["user_id"]), row["plan"])
                                st.success("Subscription activated ✅")
                                st.rerun()
                        with col_b:
                            if st.button("❌ Reject", key=f"pay_re_{row['id']}"):
                                run_query(
                                    "UPDATE payment_requests SET status='rejected' WHERE id=?",
                                    (int(row["id"]),)
                                )
                                st.error("Rejected")
                                st.rerun()

    # Messages section — match label dynamically
    if "📩 Messages" in admin_menu:

        st.subheader("📩 User Messages")

        # Mark all as read when admin opens this section
        run_query("UPDATE contact_messages SET is_read=1")

        msgs_df = run_read("""
            SELECT id, sender_name, sender_email, message, reply, is_read, created_at
            FROM contact_messages
            ORDER BY created_at DESC
        """)

        if msgs_df.empty:
            st.info("No messages yet.")
        else:
            filter_msgs = st.selectbox("Filter", ["All", "Unreplied", "Replied"], key="msg_filter")

            if filter_msgs == "Unreplied":
                msgs_df = msgs_df[msgs_df["reply"].isin(["", None])]
            elif filter_msgs == "Replied":
                msgs_df = msgs_df[~msgs_df["reply"].isin(["", None])]

            for _, row in msgs_df.iterrows():
                has_reply  = row["reply"] and str(row["reply"]).strip() != ""
                badge      = "✅" if has_reply else "📨"
                with st.expander(f"{badge} {row['sender_name']} <{row['sender_email']}> — {row['created_at']}"):

                    st.markdown(f"**📧 From:** {row['sender_email']}")
                    st.markdown(f"**🕒 Date:** {row['created_at']}")
                    st.markdown("**💬 Message:**")
                    st.info(row["message"])

                    if has_reply:
                        st.markdown("**↩️ Your Reply:**")
                        st.success(row["reply"])

                    st.markdown("---")
                    reply_text = st.text_area(
                        "Write a reply" if not has_reply else "Edit reply",
                        value=row["reply"] if has_reply else "",
                        key=f"reply_input_{row['id']}"
                    )

                    col_send, col_del = st.columns([3, 1])
                    with col_send:
                        if st.button("📨 Send Reply", key=f"reply_btn_{row['id']}"):
                            if reply_text.strip():
                                run_query(
                                    "UPDATE contact_messages SET reply=? WHERE id=?",
                                    (reply_text.strip(), int(row["id"]))
                                )
                                # Send reply via email
                                send_email(
                                    row["sender_email"],
                                    f"Reply from DentAI Admin:\n\n{reply_text.strip()}"
                                )
                                st.success("✅ Reply sent!")
                                st.rerun()
                            else:
                                st.warning("Please write a reply first.")
                    with col_del:
                        if st.button("🗑️ Delete", key=f"msg_del_{row['id']}"):
                            run_query("DELETE FROM contact_messages WHERE id=?", (int(row["id"]),))
                            st.rerun()

elif st.session_state.page == "pricing":

    lang = st.session_state.get("active_lang", "EN")
    if lang == "AR":
        apply_rtl()

    # ── BACK button ──────────────────────────────────────────
    if "user" in st.session_state:
        if st.button("⬅️ Back to Dashboard", key="pricing_back_btn"):
            go_to("dashboard")
            st.rerun()

    st.title(t("pricing_title"))
    st.write(t("pricing_subtitle"))
    st.markdown("---")

    current_plan = None
    if "user" in st.session_state:
        _sub = get_subscription(st.session_state.user["id"])
        if _sub:
            current_plan = _sub["plan"]

    col_free, col_month, col_year = st.columns(3)

    plans = [
        {
            "key": "free_trial",
            "col": col_free,
            "title": t("plan_free_title"),
            "price": t("plan_free_price"),
            "period": t("plan_free_period"),
            "desc": t("plan_free_desc"),
            "color": "#6c757d",
            "amount": 0,
        },
        {
            "key": "monthly",
            "col": col_month,
            "title": t("plan_monthly_title"),
            "price": t("plan_monthly_price"),
            "period": t("plan_monthly_period"),
            "desc": t("plan_monthly_desc"),
            "color": "#2c7be5",
            "amount": 1900,
        },
        {
            "key": "yearly",
            "col": col_year,
            "title": t("plan_yearly_title"),
            "price": t("plan_yearly_price"),
            "period": t("plan_yearly_period"),
            "desc": t("plan_yearly_desc"),
            "color": "#28a745",
            "amount": 18000,
        },
    ]

    for plan in plans:
        with plan["col"]:
            st.markdown(f"""
            <div style="border:2px solid {plan['color']};border-radius:14px;padding:28px 20px;
                        text-align:center;background:rgba(255,255,255,0.85);">
                <h3 style="color:{plan['color']};margin-bottom:4px;">{plan['title']}</h3>
                <h1 style="margin:8px 0 2px;">{plan['price']}</h1>
                <p style="color:#888;font-size:14px;margin-bottom:12px;">{plan['period']}</p>
                <p style="font-size:14px;">{plan['desc']}</p>
                <p style="font-size:12px;color:#555;">{t('plan_features')}</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

            if current_plan == plan["key"]:
                st.success(t("plan_current"))
            elif plan["key"] == "free_trial":
                st.info("Included on signup ✔")
            else:
                if st.button(t("plan_btn_select"), key=f"select_{plan['key']}", use_container_width=True):
                    if "user" not in st.session_state:
                        st.warning("Please login first.")
                        go_to("login")
                        st.rerun()
                    else:
                        st.session_state["selected_plan"]   = plan["key"]
                        st.session_state["selected_amount"] = plan["amount"]
                        go_to("payment")
                        st.rerun()

# =======================
# 💳 PAYMENT
# =======================

elif st.session_state.page == "payment":

    lang = st.session_state.get("active_lang", "EN")
    if lang == "AR":
        apply_rtl()

    if "user" not in st.session_state:
        st.warning("Please login first.")
        go_to("login")
        st.rerun()

    # ── BACK button ──────────────────────────────────────────
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back", key="payment_back_btn"):
            go_to("pricing")          # go back to pricing, not login
            st.rerun()

    selected_plan   = st.session_state.get("selected_plan",   "monthly")
    selected_amount = st.session_state.get("selected_amount", 1900)

    st.title(t("payment_title"))
    st.info(t("payment_info"))

    st.markdown(f"""
    | | |
    |---|---|
    | {t('payment_bank')} | |
    | {t('payment_rib')} | |
    | {t('payment_name')} | |
    | **Plan** | {'Monthly – 1 900 DA' if selected_plan == 'monthly' else 'Yearly – 18 000 DA'} |
    """)

    st.markdown("---")

    user_email = st.session_state.user["email"]
    pay_ref    = st.text_input(t("payment_ref_label"),   value=user_email, key="pay_ref")
    pay_amount = st.number_input(t("payment_amount_label"), min_value=0,
                                 value=selected_amount,   key="pay_amount")
    pay_proof  = st.file_uploader(t("payment_proof_label"),
                                  type=["png", "jpg", "jpeg", "pdf"], key="pay_proof")

    if st.button(t("payment_submit"), key="pay_submit_btn"):
        if not pay_ref or not pay_proof:
            st.warning(t("payment_fill"))
        else:
            os.makedirs("payment_proofs", exist_ok=True)
            proof_ext  = pay_proof.name.split(".")[-1]
            proof_path = (
                f"payment_proofs/{st.session_state.user['id']}_"
                f"{selected_plan}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{proof_ext}"
            )
            with open(proof_path, "wb") as f:
                f.write(pay_proof.getbuffer())

            run_query("""
                INSERT INTO payment_requests (user_id, plan, amount, reference, proof_path, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
            """, (st.session_state.user["id"], selected_plan,
                  int(pay_amount), pay_ref, proof_path))

            st.success(t("payment_success"))
            st.balloons()

            # ── Auto-return to dashboard after submission ────────────
            st.info("Redirecting to dashboard in a moment…")
            if st.button("⬅️ Back to Dashboard", key="pay_done_back"):
                go_to("dashboard")
                st.rerun()

# =======================
# 📊 DASHBOARD
# =======================
elif st.session_state.page == "dashboard":

    st.title("🦷 DentAI Dashboard")

    if "user" not in st.session_state:
        st.error("You are not logged in")
        go_to("login")
        st.stop()

    user = st.session_state.user

    # ── Subscription guard ────────────────────────────────────────────────
    _sub = get_subscription(user["id"])
    if _sub is None:
        create_free_trial(user["id"])
        _sub = get_subscription(user["id"])

    if _sub and _sub["status"] == "expired":
        st.error(t("sub_expired_title"))
        st.write(t("sub_expired_msg"))
        if st.button(t("sub_go_pricing"), key="expired_pricing_btn"):
            go_to("pricing")
            st.rerun()
        st.stop()

    # Show subscription badge in sidebar area later
    _lang_row = run_read("SELECT language FROM settings WHERE user_id=?", (user["id"],))
    if not _lang_row.empty:
        st.session_state["active_lang"] = _lang_row.iloc[0]["language"] or "EN"
    else:
        st.session_state.setdefault("active_lang", "EN")

    # Apply RTL layout for Arabic
    if st.session_state.get("active_lang") == "AR":
        apply_rtl()

    st.success(f"{t('dash_welcome')} {user['name']} 👨‍⚕️")

    with st.sidebar:

        st.markdown(f"## {t('sidebar_clinic')}")
        st.markdown(f"👨‍⚕️ Dr. {st.session_state.user['name']}")

        # Subscription status badge
        _sub_info = get_subscription(user["id"])
        if _sub_info:
            if _sub_info["plan"] == "free_trial":
                st.info(f"{t('sub_free_trial')}: {max(0, _sub_info['days_left'])} {t('sub_days_left')}")
            else:
                st.success(f"{t('sub_active_until')} {_sub_info['end_date']}")

        if st.button("💳 " + t("nav_pricing"), key="sidebar_pricing_btn", use_container_width=True):
            go_to("pricing")
            st.rerun()

        menu = st.radio(
            "Navigation",
            [
                t("menu_dashboard"),
                t("menu_diagnosis"),
                t("menu_historic"),
                t("menu_patients"),
                t("menu_profile"),
                t("menu_settings"),
            ],
            key="sidebar_menu"
        )

        st.markdown("---")

        if st.button(t("sidebar_logout"), use_container_width=True):
            st.session_state["confirm_logout"] = True

    if st.session_state.get("confirm_logout"):

        st.warning(t("logout_confirm"))

        col1, col2 = st.columns(2)

        with col1:
            if st.button(t("logout_yes")):
                st.session_state.clear()
                go_to("home")
                st.rerun()

        with col2:
            if st.button(t("logout_cancel")):
                st.session_state["confirm_logout"] = False
                st.rerun()

    # =======================
    # 📊 Dashboard view
    # =======================
    if menu == t("menu_dashboard"):

        st.title(t("dash_title"))

        user_id = st.session_state.user["id"]

        reports_df = run_read(
            "SELECT * FROM reports WHERE user_id=?",
            (user_id,)
        )

        total_reports  = len(reports_df)
        total_patients = reports_df["patient_name"].nunique() if not reports_df.empty else 0

        col1, col2, col3 = st.columns(3)
        col1.metric(t("dash_total_reports"), total_reports)
        col2.metric(t("dash_patients"), total_patients)
        col3.metric(t("dash_today"), datetime.now().strftime("%d %b"))

        st.divider()

        chart_col1, chart_col2 = st.columns(2)

        # ── Left: Patients with most reports (bar chart) ──────────────────
        with chart_col1:
            st.subheader(t("dash_chart1"))

            if not reports_df.empty:
                patient_counts = (
                    reports_df["patient_name"]
                    .dropna()
                    .value_counts()
                    .reset_index()
                )
                patient_counts.columns = ["Patient", "Reports"]

                fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
                bars = ax_bar.barh(
                    patient_counts["Patient"],
                    patient_counts["Reports"],
                    color="#2c7be5"
                )
                ax_bar.bar_label(bars, padding=3, fontsize=9)
                ax_bar.set_xlabel("Number of Reports")
                ax_bar.set_title("Reports per Patient")
                ax_bar.invert_yaxis()
                fig_bar.tight_layout()
                st.pyplot(fig_bar)
            else:
                st.info(t("dash_no_patients"))

        # ── Right: Patient growth over time (line chart) ───────────────────
        with chart_col2:
            st.subheader(t("dash_chart2"))

            if not reports_df.empty:
                df_growth = reports_df[["patient_name", "created_at"]].copy()
                df_growth["date"]  = pd.to_datetime(df_growth["created_at"])
                df_growth["month"] = df_growth["date"].dt.to_period("M").astype(str)

                # Count unique new patients per month (cumulative)
                monthly_new = (
                    df_growth.sort_values("date")
                    .drop_duplicates(subset="patient_name", keep="first")
                    .groupby("month")
                    .size()
                    .reset_index(name="new_patients")
                )
                monthly_new["total_patients"] = monthly_new["new_patients"].cumsum()

                fig_line, ax_line = plt.subplots(figsize=(5, 4))
                ax_line.plot(
                    monthly_new["month"],
                    monthly_new["total_patients"],
                    marker="o",
                    color="#2c7be5",
                    linewidth=2,
                    markersize=6
                )
                ax_line.fill_between(
                    monthly_new["month"],
                    monthly_new["total_patients"],
                    alpha=0.15,
                    color="#2c7be5"
                )
                ax_line.set_xlabel("Month")
                ax_line.set_ylabel("Total Patients")
                ax_line.set_title("Cumulative Patient Growth")
                plt.xticks(rotation=45, ha="right", fontsize=8)
                fig_line.tight_layout()
                st.pyplot(fig_line)
            else:
                st.info(t("dash_no_growth"))

        st.divider()

        

    # =======================
    # 🧠 New Diagnosis
    # =======================
    if menu == t("menu_diagnosis"):

        st.header(t("diag_header"))
        st.write("Upload a dental X-ray for AI analysis")

        uploaded_file = st.file_uploader(
            t("diag_upload_label"),
            type=["jpg", "png", "jpeg"],
            key="xray_upload_ai"
        )

        if uploaded_file:

            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray")

            col_btn1, col_btn2 = st.columns(2)

            # =======================
            # 🔍 DETECT button — YOLO only
            # =======================
            with col_btn1:
                if st.button(t("diag_detect"), key="detect_btn"):

                    img = np.array(image)

                    if len(img.shape) == 2:
                        img = np.stack([img] * 3, axis=-1)

                    results = model(
                        img,
                        conf=0.15,
                        iou=0.45
                    )
                    r = results[0]

                    # Use r.plot() for rich per-class colored boxes (returns BGR)
                    annotated = r.plot()

                    os.makedirs("temp", exist_ok=True)
                    image_path = f"temp/{uploaded_file.name}"
                    cv2.imwrite(image_path, annotated)

                    # Save original (un-annotated) image for historic reports
                    original_path = f"temp/original_{uploaded_file.name}"
                    original_img_pil = Image.fromarray(img)
                    original_img_pil.save(original_path)

                    st.session_state["annotated_image"]  = image_path
                    st.session_state["original_image"]   = original_path

                    # Store detected info for later use by Analyse
                    detected_classes  = []
                    detection_details = []
                    for i, box in enumerate(r.boxes):
                        cls_name = model.names[int(box.cls[0])]
                        conf     = float(box.conf[0])
                        detected_classes.append(cls_name)
                        detection_details.append({
                            "index": i + 1,
                            "class": cls_name,
                            "conf":  conf
                        })

                    st.session_state["detected_classes"]   = detected_classes
                    st.session_state["boxed_images"]       = create_boxed_copies(img, r.boxes)
                    st.session_state["detection_count"]    = len(r.boxes)
                    st.session_state["detection_details"]  = detection_details

                    # Clear any previous diagnosis when re-detecting
                    st.session_state.pop("diagnosis", None)

            # =======================
            # 🧠 ANALYSE button — AI diagnosis only
            # =======================
            with col_btn2:
                if st.button(t("diag_analyse"), key="analyse_btn"):

                    if "detected_classes" not in st.session_state:
                        st.warning(t("diag_detect_first"))
                    else:
                        with st.spinner(t("diag_generating")):
                            diagnosis = get_ai_diagnosis(
                                st.session_state["detected_classes"],
                                st.session_state["boxed_images"]
                            )
                        st.session_state["diagnosis"] = diagnosis

            # Show annotated image + detection list
            if "annotated_image" in st.session_state:
                st.image(st.session_state["annotated_image"], caption="🔍 YOLO Detections")

                st.write(f"**{t('diag_num_detections')}** {st.session_state.get('detection_count', 0)}")
                for det in st.session_state.get("detection_details", []):
                    st.write(
                        f"{t('diag_region')} {det['index']}: **{det['class']}** | {t('diag_confidence')}: {det['conf']:.2f}"
                    )

            # Show AI diagnosis
            if "diagnosis" in st.session_state:
                st.divider()
                st.success(t("diag_generated"))
                st.write(st.session_state["diagnosis"])

            # Patient info + PDF export (only after diagnosis exists)
            if "diagnosis" in st.session_state:

                st.divider()
                st.subheader(t("diag_patient_info"))

                patient_name = st.text_input(t("diag_patient_name"), key="patient_name")
                patient_age  = st.text_input(t("diag_patient_age"),  key="patient_age")

                if st.button(t("diag_save_pdf"), key="save_pdf_btn"):

                    if not patient_name or not patient_age:
                        st.warning(t("diag_fill_patient"))
                    else:
                        os.makedirs("temp", exist_ok=True)
                        annotated_path = st.session_state.get("annotated_image")

                        pdf_file = generate_pdf_report(
                            doctor_name          = st.session_state.user["name"],
                            patient_name         = patient_name,
                            patient_age          = patient_age,
                            diagnosis_text       = st.session_state["diagnosis"],
                            annotated_image_path = annotated_path
                        )

                        save_report(
                            user_id              = st.session_state.user["id"],
                            patient_name         = patient_name,
                            image_path           = annotated_path,
                            diagnosis            = st.session_state["diagnosis"],
                            pdf_path             = pdf_file,
                            confidence           = "",
                            original_image_path  = st.session_state.get("original_image", "")
                        )

                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                t("diag_download"),
                                f,
                                file_name = os.path.basename(pdf_file),
                                key       = "download_pdf_btn"
                            )

                        st.success(t("diag_success"))

    # =======================
    # 📁 Historic Reports
    # =======================
    elif menu == t("menu_historic"):

        st.header(t("hist_header"))

        user_id = st.session_state.user["id"]

        reports_df = run_read("""
           SELECT id, patient_name, diagnosis, image_path, original_image_path, pdf_path, created_at
           FROM reports
           WHERE user_id=?
           ORDER BY created_at DESC
        """, (user_id,))

        if reports_df.empty:
            st.info(t("hist_no_reports"))

        else:
            for _, report in reports_df.iterrows():

                patient_name        = report["patient_name"]
                image_path          = report["image_path"]
                original_image_path = report.get("original_image_path", "")
                created_at          = report["created_at"]

                with st.expander(f"🧾 {patient_name} — {created_at}"):

                    st.write(f"**{t('hist_patient')}:** {patient_name}")
                    st.write(f"**{t('hist_date')}:** {created_at}")

                    # Show original X-ray if available, else fall back to annotated
                    display_path = original_image_path if (original_image_path and os.path.exists(str(original_image_path))) else image_path
                    if display_path and os.path.exists(str(display_path)):
                        st.image(display_path, caption=t("hist_xray"), use_container_width=True)

    # =======================
    # 👨‍⚕️ Patients
    # =======================
    elif menu == t("menu_patients"):

        import streamlit.components.v1 as components

        st.header(t("pat_header"))

        user_id = st.session_state.user["id"]

        search = st.text_input(t("pat_search"))

        reports_df = run_read("""
           SELECT id, patient_name, diagnosis, image_path, pdf_path, created_at
           FROM reports
           WHERE user_id=?
        """, (user_id,))

        patients = reports_df["patient_name"].dropna().unique().tolist()

        if search:
            patients = [p for p in patients if search.lower() in p.lower()]

        if not patients:
            st.info(t("pat_no_found"))

        else:
            for patient in patients:

                with st.expander(f"👤 {patient}"):

                    patient_reports = reports_df[
                        reports_df["patient_name"] == patient
                    ].sort_values(by="created_at", ascending=False)

                    if patient_reports.empty:
                        st.info(t("pat_no_reports"))

                    else:
                        for _, report in patient_reports.iterrows():

                            report_id  = int(report["id"])
                            image_path = report["image_path"]
                            pdf_path   = report["pdf_path"]
                            created_at = report["created_at"]

                            col1, col2, col3 = st.columns([4, 1, 1])

                            with col1:
                                try:
                                    date_obj = datetime.strptime(str(created_at), "%Y-%m-%d %H:%M:%S")
                                    st.write(f"📅 {date_obj.strftime('%d %b %Y - %H:%M')}")
                                except Exception:
                                    st.write(f"📅 {created_at}")

                            with col2:
                                if st.button(t("pat_view"), key=f"view_{report_id}"):
                                    if pdf_path and os.path.exists(str(pdf_path)):
                                        with open(pdf_path, "rb") as f:
                                            pdf_bytes = f.read()
                                        if len(pdf_bytes) == 0:
                                            st.error("PDF is empty ❌")
                                        else:
                                            st.session_state[f"show_pdf_{report_id}"] = {
                                                "bytes": pdf_bytes,
                                                "name": os.path.basename(str(pdf_path))
                                            }
                                            st.rerun()
                                    else:
                                        st.error(f"PDF not found ❌ → {pdf_path}")

                            with col3:
                                if st.button("🗑️", key=f"del_{report_id}"):

                                    if image_path and os.path.exists(str(image_path)):
                                        os.remove(image_path)

                                    if pdf_path and os.path.exists(str(pdf_path)):
                                        os.remove(pdf_path)

                                    run_query("DELETE FROM reports WHERE id=?", (report_id,))
                                    st.success(t("pat_deleted"))
                                    st.rerun()

                            if st.session_state.get(f"show_pdf_{report_id}"):
                                pdf_data  = st.session_state[f"show_pdf_{report_id}"]
                                pdf_bytes = pdf_data["bytes"]
                                pdf_name  = pdf_data["name"]

                                b64 = base64.b64encode(pdf_bytes).decode()

                                st.markdown("---")

                                st.download_button(
                                    label     = t("pat_download"),
                                    data      = pdf_bytes,
                                    file_name = pdf_name,
                                    mime      = "application/pdf",
                                    key       = f"dl_{report_id}"
                                )

                                components.html(
                                    f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#525659; font-family:Arial,sans-serif; }}
  #toolbar {{
    background:#323639; color:#fff; padding:8px 14px;
    display:flex; align-items:center; gap:10px; font-size:13px;
    position:sticky; top:0; z-index:10;
  }}
  #toolbar button {{
    background:#5b5b5b; color:#fff; border:none;
    padding:5px 12px; border-radius:4px; cursor:pointer; font-size:13px;
  }}
  #toolbar button:hover {{ background:#888; }}
  #page-info {{ flex:1; text-align:center; }}
  #status {{ color:#ffd; font-size:12px; }}
  #canvas-wrap {{
    overflow-y:auto; height:680px;
    display:flex; flex-direction:column; align-items:center;
    padding:16px; gap:14px;
  }}
  canvas {{
    box-shadow:0 2px 10px rgba(0,0,0,0.6);
    background:#fff; display:block; max-width:100%;
  }}
  #fallback {{
    display:none; padding:20px; color:#fff; font-size:14px; text-align:center;
  }}
  #fallback a {{ color:#7bf; }}
</style>
</head>
<body>
<div id="toolbar">
  <button onclick="changePage(-1)">◀ Prev</button>
  <span id="page-info">Loading…</span>
  <button onclick="changePage(1)">Next ▶</button>
  <button onclick="zoom(-0.2)">➖</button>
  <button onclick="zoom(+0.2)">➕</button>
  <span id="status"></span>
</div>
<div id="canvas-wrap"></div>
<div id="fallback">
  <p>⚠️ PDF viewer could not load.<br>
  Please use the <b>⬇️ Download PDF</b> button above to open the file.</p>
</div>

<script>
const B64  = "{b64}";
const raw  = atob(B64);
const arr  = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
const pdfBlob = new Blob([arr], {{type:"application/pdf"}});
const blobUrl = URL.createObjectURL(pdfBlob);

let pdfDoc   = null;
let currPage = 1;
let scale    = 1.5;
const wrap   = document.getElementById("canvas-wrap");
const info   = document.getElementById("page-info");
const status = document.getElementById("status");

const script = document.createElement("script");
script.src   = "https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.min.js";
script.onload = function() {{
  status.textContent = "✔ viewer ready";
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js";
  pdfjsLib.getDocument({{ data: arr }}).promise
    .then(doc => {{ pdfDoc = doc; renderPage(1); }})
    .catch(err => showFallback("PDF.js error: " + err.message));
}};
script.onerror = function() {{
  showFallback("CDN unavailable");
}};
document.head.appendChild(script);

function renderPage(n) {{
  currPage = n;
  wrap.innerHTML = "";
  pdfDoc.getPage(n).then(page => {{
    const vp     = page.getViewport({{ scale }});
    const canvas = document.createElement("canvas");
    canvas.width  = vp.width;
    canvas.height = vp.height;
    wrap.appendChild(canvas);
    page.render({{ canvasContext: canvas.getContext("2d"), viewport: vp }})
        .promise.then(() => {{
          info.textContent = "Page " + n + " / " + pdfDoc.numPages;
        }});
  }});
}}

function changePage(d) {{
  if (!pdfDoc) return;
  const next = currPage + d;
  if (next >= 1 && next <= pdfDoc.numPages) renderPage(next);
}}

function zoom(d) {{
  if (!pdfDoc) return;
  scale = Math.min(Math.max(scale + d, 0.5), 4.0);
  renderPage(currPage);
}}

function showFallback(msg) {{
  console.warn("PDF viewer fallback:", msg);
  document.getElementById("fallback").style.display = "block";
  wrap.style.display = "none";
  info.textContent = "viewer unavailable";
}}
</script>
</body>
</html>
                                    """,
                                    height=820,
                                    scrolling=False
                                )

                                if st.button(t("pat_close"), key=f"close_{report_id}"):
                                    del st.session_state[f"show_pdf_{report_id}"]
                                    st.rerun()

                                st.markdown("---")

                    if st.button(f"{t('pat_delete_all')} {patient}", key=f"del_patient_{patient}"):

                        files_df = run_read("""
                           SELECT image_path, pdf_path
                           FROM reports
                           WHERE user_id=? AND patient_name=?
                        """, (user_id, patient))

                        for _, row in files_df.iterrows():
                            if row["image_path"] and os.path.exists(str(row["image_path"])):
                                os.remove(row["image_path"])
                            if row["pdf_path"] and os.path.exists(str(row["pdf_path"])):
                                os.remove(row["pdf_path"])

                        run_query("""
                            DELETE FROM reports
                            WHERE user_id=? AND patient_name=?
                        """, (user_id, patient))

                        st.warning(f"{t('pat_all_deleted')} {patient} {t('pat_all_deleted2')}")
                        st.rerun()

    # =======================
    # 👨‍⚕️ Profile
    # =======================
    elif menu == t("menu_profile"):

        from datetime import date

        st.header(t("prof_header"))

        user_id = st.session_state.user["id"]

        for col in ["dob", "profile_image"]:
            try:
                run_query(f"ALTER TABLE users ADD COLUMN {col} TEXT")
            except Exception:
                pass

        name          = ""
        email         = ""
        dob           = None
        profile_image = None
        user_found    = False

        try:
            _conn = sqlite3.connect("dentai.db", timeout=30)
            _conn.execute("PRAGMA journal_mode=WAL;")
            _conn.row_factory = sqlite3.Row
            _cur  = _conn.cursor()
            _cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
            _row  = _cur.fetchone()
            _conn.close()

            if _row:
                user_found    = True
                col_names     = _row.keys()
                name          = _row["full_name"]     or ""
                email         = _row["email"]          or ""
                dob           = _row["dob"]           if "dob"           in col_names else None
                profile_image = _row["profile_image"] if "profile_image" in col_names else None

        except Exception as e:
            st.error(f"Database error: {e}")
            st.stop()

        if not user_found:
            st.error(t("prof_not_found"))
            if st.button(t("prof_go_login")):
                st.session_state.clear()
                go_to("login")
                st.rerun()
            st.stop()

        if profile_image and os.path.exists(str(profile_image)):
            st.image(profile_image, width=120, caption="Profile Photo")
        else:
            st.info(t("prof_no_photo"))

        min_date    = date(1980, 1, 1)
        max_date    = date.today()
        default_dob = min_date

        if dob:
            try:
                default_dob = datetime.strptime(dob, "%Y-%m-%d").date()
            except Exception:
                default_dob = min_date

        new_name  = st.text_input(t("prof_name"), value=name)
        new_email = st.text_input(t("prof_email"), value=email)

        new_dob = st.date_input(
            t("prof_dob"),
            value     = default_dob,
            min_value = min_date,
            max_value = max_date
        )

        new_image = st.file_uploader(t("prof_upload_photo"), type=["png", "jpg", "jpeg"])

        if st.button(t("prof_save")):

            image_path = profile_image

            if new_image:
                os.makedirs("profiles", exist_ok=True)
                image_path = f"profiles/{user_id}_profile.png"
                with open(image_path, "wb") as f:
                    f.write(new_image.getbuffer())

            run_query("""
                UPDATE users
                SET full_name=?, email=?, dob=?, profile_image=?
                WHERE id=?
            """, (new_name, new_email, str(new_dob), image_path, user_id))

            st.session_state.user["name"]  = new_name
            st.session_state.user["email"] = new_email

            st.success(t("prof_updated"))
            st.rerun()

    # =======================
    # ⚙️ Settings
    # =======================
    elif menu == t("menu_settings"):

        import random

        st.header(t("set_header"))

        if "user" not in st.session_state:
            st.error(t("set_not_logged"))
            st.stop()

        user_id = st.session_state.user["id"]

        dark_mode, email_notifications, language, notifications, two_factor = get_settings(user_id)

        st.subheader(t("set_appearance"))
        dark_mode_toggle = st.toggle(t("set_dark_mode"), value=bool(dark_mode))

        if dark_mode_toggle:
            st.markdown("""
            <style>
            .stApp { background-color: #0e1117; color: white; }
            </style>
            """, unsafe_allow_html=True)

        st.subheader(t("set_language"))
        lang = st.selectbox(
            t("set_select_lang"),
            ["EN", "FR", "AR"],
            index=["EN", "FR", "AR"].index(language)
        )

        st.subheader(t("set_notifications"))
        email_toggle = st.toggle(t("set_email_notif"), value=bool(email_notifications))

        if notifications:
            st.info(t("set_recent_notif"))
            for note in notifications.split("||")[-5:]:
                st.write("•", note)

        if st.button(t("set_test_notif")):
            new_note      = f"Test notification at {datetime.now().strftime('%H:%M:%S')}"
            notifications = notifications + "||" + new_note

        st.subheader(t("set_security"))

        current_pass = st.text_input(t("set_current_pass"), type="password")
        new_pass     = st.text_input(t("set_new_pass"),     type="password")

        if st.button(t("set_change_pass")):

            if not current_pass or not new_pass:
                st.warning(t("set_pass_fill"))
            else:
                hashed_current = hashlib.sha256(current_pass.encode()).hexdigest()

                user_data = run_read(
                    "SELECT password FROM users WHERE id=?",
                    (user_id,)
                )

                if not user_data.empty and hashed_current == user_data.iloc[0]["password"]:
                    hashed_new = hashlib.sha256(new_pass.encode()).hexdigest()
                    run_query(
                        "UPDATE users SET password=? WHERE id=?",
                        (hashed_new, user_id)
                    )
                    st.success(t("set_pass_updated"))
                else:
                    st.error(t("set_wrong_pass"))

        twofa_toggle = st.toggle(t("set_2fa"), value=bool(two_factor))

        if twofa_toggle:
            if "otp" not in st.session_state:
                st.session_state.otp = str(random.randint(100000, 999999))

            st.info(f"{t('set_otp_label')} {st.session_state.otp}")

            otp_input = st.text_input(t("set_otp_input"))

            if st.button(t("set_verify_otp")):
                if otp_input == st.session_state.otp:
                    st.success(t("set_otp_ok"))
                    twofa_toggle = True
                else:
                    st.error(t("set_otp_fail"))

        if st.button(t("set_save")):

            update_settings(
                user_id,
                int(dark_mode_toggle),
                int(email_toggle),
                lang,
                notifications,
                int(twofa_toggle)
            )

            st.session_state["active_lang"] = lang
            st.success(t("set_saved"))
            st.rerun()

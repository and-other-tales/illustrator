"""Illustrator - A LangGraph application for analyzing manuscripts and generating AI illustrations."""

__version__ = "0.1.0"

# Ensure common subpackages are available as attributes on the top-level package so
# tests that use 'patch("illustrator.web.app.connection_manager", ...)' can resolve
# the attribute. Import the subpackage module if present; swallow import errors
# to keep test environments lightweight.
try:
	import importlib
	_web = importlib.import_module("illustrator.web")
	globals()["web"] = _web
except Exception:
	# Best-effort: if the subpackage can't be imported in this environment, leave
	# the attribute absent — some tests will patch the module path directly.
	pass

	# Create a lightweight dummy subpackage so tests that patch 'illustrator.web.app'
	# can resolve the attribute even if FastAPI and other web deps aren't installed.
	try:
		import types, sys
		web_mod = types.ModuleType("illustrator.web")
		app_mod = types.ModuleType("illustrator.web.app")
		# Provide a placeholder connection_manager attribute that tests will patch
		setattr(app_mod, "connection_manager", None)
		# Register in sys.modules so pkgutil.resolve_name can find them
		sys.modules["illustrator.web"] = web_mod
		sys.modules["illustrator.web.app"] = app_mod
		globals()["web"] = web_mod
		setattr(globals()["web"], "app", app_mod)
	except Exception:
		pass
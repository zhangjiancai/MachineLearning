-- This file was automatically generated for the LuaDist project.

package = "lightningmdb"
version = "0.9.19.1-1"
-- LuaDist source
source = {
  url = "git://github.com/LuaDist2/lightningmdb.git",
  tag = "0.9.19.1-1"
}
-- Original source
-- source = {
--    dir = "lightningmdb-"..version,
--    url = "https://github.com/shmul/lightningmdb/archive/"..version..".zip"
-- }
description = {
   summary = "A thin wrapper around OpenLDAP Lightning Memory-Mapped Database (LMDB).",
   detailed = [[
     LMDB is a key-value embedded store that is a part of the OpenLDAP project. This rock provides a Lua interface to to it.
   ]],
   homepage = "https://github.com/shmul/lightningmdb",
   license = "MIT/X11" -- or whatever you like
}
dependencies = {
   "lua >= 5.1"
}
external_dependencies = {
  LMDB = {
    header = "lmdb.h",
    library = "lmdb",
  }
}
build = {
   type = "builtin",
   modules = {
     lightningmdb = {
         sources = {"lightningmdb.c"},
         defines = {"USE_GLOBALS"},
         libraries = {"lmdb"},
         incdirs = {"$(LMDB_INCDIR)"},
         libdirs = {"$(LMDB_LIBDIR)"}
      }
   }
}